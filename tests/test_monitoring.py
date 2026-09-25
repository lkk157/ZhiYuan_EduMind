# -*- coding: utf-8 -*-
"""
M6 单测：数据看板——聚合算术、近 7 天窗口、sources 聚合与容错、鉴权与越权隔离。

为什么直接 db.add 造消息而不用 crud.append_message_pair：
趋势分桶依赖 created_at 的「天」，而 server_default 的库时钟在 SQLite 是 UTC、
MySQL 是本地时区，测试不能赌——显式传 created_at + 注入 now 时钟，分桶断言才可复现。
"""
import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.db import crud
from app.db.models import Message
from app.db.session import get_db

# 固定「现在」：趋势断言与造数据共用一个时钟，跨午夜跑测试结果不变
NOW = datetime(2026, 9, 25, 12, 0, 0)


@pytest.fixture()
def env(db_session):
    """SQLite 会话 + A/B 两用户 + 各自的会话与统计样本数据。"""
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db

    user_a = crud.create_user(db_session, username="alice", password_hash="x")
    user_b = crud.create_user(db_session, username="bob", password_hash="x")
    conv_a = crud.create_conversation(db_session, user_id=user_a.id, title="A的会话")
    conv_b = crud.create_conversation(db_session, user_id=user_b.id, title="B的会话")

    def add_pair(conv, *, question, hit, sources, created_at):
        """显式时间的一问一答（看板只读库，不经过 API——造数据少走一层）。"""
        db_session.add(
            Message(
                conversation_id=conv.id, role="user", content=question,
                hit=None, sources="[]", created_at=created_at,
            )
        )
        db_session.add(
            Message(
                conversation_id=conv.id, role="assistant", content="回答",
                hit=hit, sources=sources, created_at=created_at,
            )
        )
        db_session.commit()

    # A：5 轮问答——今天 3（2 命中 1 兜底）+ 3 天前 1 兜底 + 10 天前 1 命中（窗口外）
    src_docx = json.dumps([{"file_name": "讲义.docx", "page_no": 1}])
    src_two = json.dumps([
        {"file_name": "讲义.docx", "page_no": 2},
        {"file_name": "课件B.pdf", "page_no": 5},
    ])
    add_pair(conv_a, question="学习率过大会怎样", hit=True, sources=src_two, created_at=NOW)
    add_pair(conv_a, question="什么是三要素", hit=True, sources=src_docx, created_at=NOW)
    add_pair(conv_a, question="随便问问", hit=False, sources="[]",
             created_at=NOW - timedelta(days=3))
    # 10 天前：进 total/命中率/高频，不进 last7 与 trend
    add_pair(conv_a, question="旧问题", hit=True,
             sources=json.dumps([{"file_name": "旧课件.pdf", "page_no": 9}]),
             created_at=NOW - timedelta(days=10))
    # 脏 sources 行：坏 JSON 不得让看板 500
    add_pair(conv_a, question="脏数据", hit=True, sources="not json", created_at=NOW)

    # B：1 轮命中问答 + 1 条答错记录（用于越权对账）
    add_pair(conv_b, question="B的问题", hit=True,
             sources=json.dumps([{"file_name": "B课件.pdf", "page_no": 1}]), created_at=NOW)
    crud.create_quiz_record(
        db_session, user_id=user_b.id, question="B错题", question_type="choice",
        user_answer="A", correct_answer="B", explanation="", is_correct=False,
    )

    # A：4 条答题（3 对 1 错）+ 1 组 2 文档共 10 块
    for i in range(3):
        crud.create_quiz_record(
            db_session, user_id=user_a.id, question=f"题{i}", question_type="choice",
            user_answer="A", correct_answer="A", explanation="", is_correct=True,
        )
    crud.create_quiz_record(
        db_session, user_id=user_a.id, question="错题", question_type="choice",
        user_answer="B", correct_answer="A", explanation="", is_correct=False,
    )
    group = crud.create_kb_group(db_session, user_id=user_a.id, name="数据结构")
    for name, chunks in (("讲义.docx", 7), ("课件B.pdf", 3)):
        crud.create_document(
            db_session, user_id=user_a.id, group_id=group.id, file_name=name,
            file_path=f"data/uploads/{name}", file_hash="h", chunk_count=chunks,
        )

    token_a = create_access_token(user_id=user_a.id, username=user_a.username)
    token_b = create_access_token(user_id=user_b.id, username=user_b.username)
    client = TestClient(app)  # 裸构造：不触发 startup（不连真 MySQL）
    yield {
        "client": client,
        "db": db_session,
        "token_a": token_a,
        "token_b": token_b,
        "user_a": user_a,
        "conv_a": conv_a,
    }
    app.dependency_overrides.clear()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_overview_aggregates_own_data(env):
    """A 视角全量对账：提问/命中/高频/答题/知识库逐字段与造数一致（含脏 JSON 容错）。"""
    from app.monitoring import stats

    # 趋势用注入时钟固定 NOW，其余口径全量——直接调统计层算
    ov = stats.overview(env["db"], user_id=env["user_a"].id, now=NOW)
    assert ov["questions"]["total"] == 5
    assert ov["questions"]["last7"] == 4  # 10 天前那轮被窗口排除
    assert ov["answers"] == {"total": 5, "hit": 4, "fallback": 1, "hit_rate": 0.8}
    # 高频：讲义 2 次 > 其余各 1 次；同数按文件名升序——「旧」(U+65E7) < 「课」(U+8BFE)
    # （首跑即栽在这：断言想当然按语义顺序写，实现的确定性字典序才是可复现契约）
    # 坏 JSON 行不贡献计数、也不让统计崩
    assert [(x["file_name"], x["count"]) for x in ov["top_files"]] == [
        ("讲义.docx", 2),
        ("旧课件.pdf", 1),
        ("课件B.pdf", 1),
    ]
    assert ov["quiz"] == {"total": 4, "correct": 3, "wrong": 1, "accuracy": 0.75}
    assert ov["kb"] == {"groups": 1, "documents": 2, "chunks": 10}


def test_overview_trend_window_and_zero_padding(env):
    """趋势恒 7 项（今天收尾、缺桶补 0）；3 天前计数在、10 天前不进窗口。"""
    from app.monitoring import stats

    trend = stats.overview(env["db"], user_id=env["user_a"].id, now=NOW)["questions"]["trend"]
    assert len(trend) == 7
    assert trend[-1] == {"date": "09-25", "count": 3}  # 今天 3 轮用户消息
    assert trend[-4] == {"date": "09-22", "count": 1}  # 3 天前
    assert trend[0] == {"date": "09-19", "count": 0}  # 首日无提问补 0，x 轴不断档
    assert all(item["count"] >= 0 for item in trend)


def test_overview_endpoint_auth_and_isolation(env):
    """接口契约：未登录 401；B 只见 B 的数据（一切统计过 user_id，越权纪律同款）。"""
    c = env["client"]
    assert c.get("/monitoring/overview").status_code == 401

    ov_b = c.get("/monitoring/overview", headers=_auth(env["token_b"])).json()
    assert ov_b["questions"]["total"] == 1  # A 的 5 轮一条都不串进来
    assert ov_b["answers"]["hit_rate"] == 1.0
    assert ov_b["quiz"] == {"total": 1, "correct": 0, "wrong": 1, "accuracy": 0.0}
    assert [x["file_name"] for x in ov_b["top_files"]] == ["B课件.pdf"]
    assert ov_b["kb"]["groups"] == 0  # B 没建组


def test_overview_zero_data_does_not_crash(db_session):
    """全新用户零数据：全 0 返回、命中率 0.0（除零保护）、趋势 7 项补 0——看板不 500。"""
    from app.main import app
    from app.monitoring import stats

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    try:
        fresh = crud.create_user(db_session, username="fresh", password_hash="x")
        ov = stats.overview(db_session, user_id=fresh.id, now=NOW)
        assert ov["questions"] == {
            "total": 0,
            "last7": 0,
            "trend": [{"date": f"{m:02d}-{d:02d}", "count": 0} for m, d in
                      [(9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25)]],
        }
        assert ov["answers"]["hit_rate"] == 0.0
        assert ov["quiz"]["accuracy"] == 0.0
        assert ov["top_files"] == []
    finally:
        app.dependency_overrides.clear()
