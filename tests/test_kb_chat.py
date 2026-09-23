# -*- coding: utf-8 -*-
"""
接口层单测（M2）：/kb/* 与 /chat/ask 的行为契约——增量入库可视化、防幻觉硬闸门、越权 404。

为什么用 TestClient 但不进上下文管理器：
`with TestClient(app)` 会触发 startup 的 init_db() 去连真实 MySQL（污染真库）；
裸构造 + dependency_overrides 注入 SQLite 会话即可测路由行为，零外部依赖。
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import create_access_token
from app.db import crud
from app.db.session import get_db
from app.rag import embeddings
from app.rag.prompts import FALLBACK_MESSAGE
from app.rag.vector_store import ChromaStore

DOC_TEXT = "梯度下降的学习率过大会导致损失函数震荡不收敛。"


def _docx_bytes(paras: list[str]) -> bytes:
    """python-docx 现造 docx 字节流（单测不落盘临时文件，直接喂 multipart）。"""
    from docx import Document

    doc = Document()
    for p in paras:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _patch_store_and_embed(monkeypatch, chroma_client, fake_embed_fn):
    """把 store_for 全部缝点 + 向量化缝换成内存版（四个命名空间都补，覆盖两种补丁习惯）。"""

    def fake_store_for(user_id, group_id, client=None):
        return ChromaStore(collection=f"u{user_id}g{group_id}", client=chroma_client)

    monkeypatch.setattr("app.rag.vector_store.store_for", fake_store_for)
    monkeypatch.setattr("app.rag.retriever.store_for", fake_store_for)
    monkeypatch.setattr("app.ingest.pipeline.store_for", fake_store_for)
    monkeypatch.setattr("app.api.kb.store_for", fake_store_for)
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)


class _StubGateway:
    """假 LLM 网关：返回固定文本并计数调用（防幻觉断言靠它）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return self.reply


class _NoCallGateway(_StubGateway):
    """一旦被调用就让测试失败——专用于「严禁调 LLM」分支的硬校验。"""

    async def generate(self, **kwargs):
        raise AssertionError("防幻觉硬闸门失效：该分支严禁调用 LLM")


@pytest.fixture()
def env(db_session, monkeypatch, chroma_client, fake_embed_fn):
    """注入 SQLite 会话 + 内存向量库/假向量，并造好 A、B 两个用户。"""
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    _patch_store_and_embed(monkeypatch, chroma_client, fake_embed_fn)

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
    }
    app.dependency_overrides.clear()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===== 分组 CRUD =====


def test_group_crud_and_duplicate_409(env):
    """建组 201 / 列表含 doc_count / 重名 409 人话 / 删除 ok。"""
    c = env["client"]
    r = c.post("/kb/groups", json={"name": "高等数学"}, headers=_auth(env["token_a"]))
    assert r.status_code == 201
    gid = r.json()["id"]

    r = c.get("/kb/groups", headers=_auth(env["token_a"]))
    assert r.status_code == 200
    groups = r.json()
    assert groups[0]["name"] == "高等数学"
    assert groups[0]["doc_count"] == 0

    r = c.post("/kb/groups", json={"name": "高等数学"}, headers=_auth(env["token_a"]))
    assert r.status_code == 409
    assert "分组名已存在" in r.json()["error"]["message"]

    r = c.delete(f"/kb/groups/{gid}", headers=_auth(env["token_a"]))
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_validation_error_uses_unified_format(env):
    """422 也走统一错误出口（前端只写一个解析函数——M1 立的契约延续）。"""
    r = env["client"].post("/kb/groups", json={"name": ""}, headers=_auth(env["token_a"]))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == 422


# ===== 上传与增量 =====


def test_upload_incremental_and_skip_and_change(env):
    """上传→增量计数；同名同内容重传 skipped_identical；改内容重传 changed>=1。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    data = _docx_bytes([DOC_TEXT])

    r = c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", data)}, headers=_auth(env["token_a"]))
    assert r.status_code == 200
    result = r.json()
    assert result["skipped_identical"] is False
    assert result["added"] >= 1
    assert result["page_count"] == 1
    assert result["empty_pages"] == []

    docs = c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_a"])).json()
    assert docs[0]["file_name"] == "讲义.docx"
    assert docs[0]["status"] == "ready"
    assert docs[0]["chunk_count"] == result["chunk_count"]

    # 同名同内容重传：文件级指纹短路
    r = c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", data)}, headers=_auth(env["token_a"]))
    assert r.json()["skipped_identical"] is True

    # 同名改内容重传：chunk 级增量（0 号块 hash 变了 → changed）
    data2 = _docx_bytes([DOC_TEXT + "补充：应使用学习率衰减。"])
    r = c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", data2)}, headers=_auth(env["token_a"]))
    result = r.json()
    assert result["skipped_identical"] is False
    assert result["changed"] + result["added"] >= 1


# ===== 问答：命中 / 兜底 =====


def test_ask_hit_appends_real_sources_and_strips_fake(env, monkeypatch):
    """命中：强制拼接真实【来源】；模型自写的伪来源被删（三层防线合成验证）。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    stub = _StubGateway("学习率过大会震荡。【来源：编造.pdf，第99页】")
    monkeypatch.setattr("app.api.chat.gateway", stub)

    r = c.post(
        "/chat/ask",
        json={"question": "梯度下降的学习率过大会导致什么", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hit"] is True
    assert stub.calls == 1
    assert "编造" not in body["answer"]  # 伪来源被 sanitize 删掉
    assert "来源：" in body["answer"]  # 真来源被强制拼接
    assert "讲义.docx" in body["answer"]
    assert body["sources"][0]["file_name"] == "讲义.docx"
    assert body["sources"][0]["page_no"] == 1


def test_ask_fallback_never_calls_llm(env, monkeypatch):
    """★ 防幻觉硬闸门：空检索直接兜底，LLM 一旦被调用测试即失败。"""
    c = env["client"]

    async def fake_retrieve(question, **kwargs):
        return []

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.chat.gateway", _NoCallGateway(""))

    r = c.post("/chat/ask", json={"question": "任意问题"}, headers=_auth(env["token_a"]))
    body = r.json()
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE
    assert body["sources"] == []


def test_ask_unrelated_question_falls_back(env, monkeypatch):
    """高阈值下无关问题走兜底（真检索链路 + generate 不被调用）。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    monkeypatch.setattr(settings, "score_threshold", 0.99)  # 近乎苛刻：非同文必被滤掉
    monkeypatch.setattr("app.api.chat.gateway", _NoCallGateway(""))
    r = c.post(
        "/chat/ask",
        json={"question": "光合作用的暗反应发生在叶绿体基质吗", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    body = r.json()
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE


# ===== 越权防线 =====


def test_cross_user_access_is_404(env):
    """B 操作 A 的分组/文档一律 404（不泄漏存在性）；借 A 的分组提问也 404。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "A的组"}, headers=_auth(env["token_a"])).json()["id"]
    did = c.post(
        f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"])
    ).json()["doc_id"]

    r = c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_b"]))
    assert r.status_code == 404
    assert "error" in r.json()

    r = c.delete(f"/kb/documents/{did}", headers=_auth(env["token_b"]))
    assert r.status_code == 404

    r = c.post("/chat/ask", json={"question": "梯度下降", "group_ids": [gid]}, headers=_auth(env["token_b"]))
    assert r.status_code == 404

    # 对照：A 自己访问正常
    assert c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_a"])).status_code == 200
