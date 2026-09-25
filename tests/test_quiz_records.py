# -*- coding: utf-8 -*-
"""
M5 单测：错题落库链路——判分即记、教材锚、错题本读写删、越权 404。

为什么全用单选卷：单选判分在代码层零 LLM（能确定的绝不交给模型），
本组用例测的是「落库与权限」而不是「LLM 判得准不准」（后者见 test_agent_tools）。
"""
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.db import crud
from app.db.session import get_db
from app.rag import embeddings
from app.rag.vector_store import ChromaStore


@pytest.fixture()
def env(db_session, monkeypatch, chroma_client, fake_embed_fn):
    """SQLite + 内存向量库 + 记忆索引打桩 + A/B 两用户。"""
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db

    # 知识库向量缝（与 test_kb_chat 同款四缝）
    def fake_store_for(user_id, group_id, client=None):
        return ChromaStore(collection=f"u{user_id}g{group_id}", client=chroma_client)

    for ns in ("app.rag.vector_store", "app.rag.retriever", "app.ingest.pipeline", "app.api.kb"):
        monkeypatch.setattr(f"{ns}.store_for", fake_store_for)
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    # 记忆双写的索引侧打桩（绝不能打到真实磁盘 chroma）
    from app.rag.vector_store import memory_store_for as _real_msf  # noqa: F401

    class _MemStore:
        def __init__(self):
            self.upserts = []

        async def upsert(self, *, ids, texts, metadatas):
            self.upserts.append((list(ids), list(texts)))

        def query(self, vector, top_k):
            return []

    mem = _MemStore()
    monkeypatch.setattr("app.memory.long_term.memory_store_for", lambda uid: mem)

    user_a = crud.create_user(db_session, username="alice", password_hash="x")
    user_b = crud.create_user(db_session, username="bob", password_hash="x")
    token_a = create_access_token(user_id=user_a.id, username=user_a.username)
    token_b = create_access_token(user_id=user_b.id, username=user_b.username)
    client = TestClient(app)  # 裸构造：不触发 startup（不连真 MySQL）
    yield {
        "client": client,
        "db": db_session,
        "token_a": token_a,
        "token_b": token_b,
        "user_a": user_a,
        "user_b": user_b,
        "mem": mem,
    }
    app.dependency_overrides.clear()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


_CHOICE_QUIZ = {
    "type": "quiz",
    "questions": [
        {
            "type": "choice",
            "question": "单链表插入操作的时间复杂度是？",
            "options": ["A. O(1)", "B. O(n)", "C. O(log n)", "D. O(n^2)"],
            "answer": "B",
            "explanation": "需要先遍历到前驱节点。",
        },
        {
            "type": "choice",
            "question": "顺序表支持随机访问吗？",
            "options": ["A. 支持", "B. 不支持"],
            "answer": "A",
            "explanation": "按下标直接定位。",
        },
    ],
}


def test_score_persists_records_with_anchor_and_memory(env):
    """判分即落库：答错进错题本（带教材锚）+ 答对也记录（周报要算正确率）+ 薄弱记忆双写。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    r = c.post(
        "/chat/score",
        json={
            "quiz": _CHOICE_QUIZ,
            "answers": ["A", "A"],  # 第1题错（应B），第2题对
            "sources": [{"file_name": "2.3.1_单链表的定义.pdf", "page_no": 3}],
        },
        headers=tok,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 50
    assert body["recorded"] == 1  # 只有错题数进提示

    # 错题本只列错的，且带教材锚与图谱推荐字段
    records = c.get("/quiz/records", headers=tok).json()
    assert len(records) == 1
    rec = records[0]
    assert "单链表插入" in rec["question"]
    assert rec["user_answer"] == "A" and rec["correct_answer"] == "B"
    assert rec["ref_file"] == "2.3.1_单链表的定义.pdf"
    assert rec["ref_page"] == 3
    assert isinstance(rec["related"], list)  # 图谱推荐字段存在（结构边是否非空取决于课件清单）

    # 全部作答（含对的）已入 quiz_records（周报正确率的数据源）；薄弱记忆恰一条
    all_rows = crud.list_quiz_records(env["db"], user_id=env["user_a"].id, wrong_only=False)
    assert len(all_rows) == 2
    assert sum(1 for x in all_rows if x.is_correct) == 1
    facts = crud.list_memory_facts(env["db"], user_id=env["user_a"].id)
    assert len(facts) == 1 and facts[0].kind == "weak_point"
    assert len(env["mem"].upserts) == 1  # 双写的索引侧也写了一次


def test_records_delete_and_cross_user_404(env):
    """错题读写删全带 user_id：B 看不到 A 的错题、删 A 的报 404；A 删除后列表空。"""
    c = env["client"]
    tok_a, tok_b = _auth(env["token_a"]), _auth(env["token_b"])
    c.post(
        "/chat/score",
        json={"quiz": _CHOICE_QUIZ, "answers": ["A", "B"], "sources": []},  # 两题全错
        headers=tok_a,
    )
    assert len(c.get("/quiz/records", headers=tok_a).json()) == 2
    assert c.get("/quiz/records", headers=tok_b).json() == []  # B 一无所见

    rid = c.get("/quiz/records", headers=tok_a).json()[0]["id"]
    assert c.delete(f"/quiz/records/{rid}", headers=tok_b).status_code == 404  # 越权删
    assert c.delete(f"/quiz/records/{rid}", headers=tok_a).status_code == 200
    assert len(c.get("/quiz/records", headers=tok_a).json()) == 1
    assert c.delete(f"/quiz/records/{rid}", headers=tok_a).status_code == 404  # 幂等 404


def test_score_invalid_quiz_400(env):
    """quiz 结构不对 → 400 人话；解析没通过绝不落库（没判明白就不记录）。"""
    c = env["client"]
    r = c.post(
        "/chat/score",
        json={"quiz": {"questions": []}, "answers": []},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 400
    assert crud.list_quiz_records(env["db"], user_id=env["user_a"].id, wrong_only=False) == []
