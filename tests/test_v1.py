# -*- coding: utf-8 -*-
"""
M7-2 单测：/v1 服务化网关——API Key 鉴权、账号隔离、检索/问答/流式契约、429 背压。

为什么复用 test_kb_chat 的「四缝 + 假向量」模式：
/v1 的业务实现就是既有函数（ingest_uploads / retrieve / run_agent），
本组只验证「服务化契约与鉴权边界」，不重复测检索/生成本身的正确性。
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import create_access_token
from app.db import crud
from app.db.session import get_db
from app.rag import embeddings
from app.rag.vector_store import ChromaStore

DOC_TEXT = "梯度下降的学习率过大会导致损失函数震荡不收敛。"


def _docx_bytes(paras: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for p in paras:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class _StubGateway:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.kwargs: dict = {}

    async def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.reply

    async def generate_stream(self, **kwargs):
        """SSE 路径专用：on_delta 传入时 tools 走流式网关方法（逐字回吐）。"""
        self.calls += 1
        self.kwargs = kwargs
        for piece in self.reply:
            yield piece


@pytest.fixture()
def env(db_session, monkeypatch, chroma_client, fake_embed_fn):
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db

    # 知识库向量缝（与 test_kb_chat 同款）
    def fake_store_for(user_id, group_id, client=None):
        return ChromaStore(collection=f"u{user_id}g{group_id}", client=chroma_client)

    for ns in ("app.rag.vector_store", "app.rag.retriever", "app.ingest.pipeline", "app.api.kb"):
        monkeypatch.setattr(f"{ns}.store_for", fake_store_for)
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)

    # 记忆索引侧（v1 ask 会召回记忆；空实现保证零磁盘）
    class _MemStore:
        async def upsert(self, **kwargs):
            pass

        def query(self, vector, top_k):
            return []

    monkeypatch.setattr("app.memory.long_term.memory_store_for", lambda uid: _MemStore())

    user_a = crud.create_user(db_session, username="alice", password_hash="x")
    user_b = crud.create_user(db_session, username="bob", password_hash="x")
    group_a = crud.create_kb_group(db_session, user_id=user_a.id, name="A组")
    group_b = crud.create_kb_group(db_session, user_id=user_b.id, name="B组")
    _, key_a = crud.create_api_key(db_session, user_id=user_a.id, name="portal")
    _, key_b = crud.create_api_key(db_session, user_id=user_b.id, name="b")
    client = TestClient(app)  # 裸构造：不触发 startup
    yield {
        "client": client,
        "db": db_session,
        "user_a": user_a,
        "group_a": group_a,
        "group_b": group_b,
        "key_a": key_a,
        "key_b": key_b,
        "token_a": create_access_token(user_id=user_a.id, username=user_a.username),
    }
    app.dependency_overrides.clear()


def _key(k: str) -> dict:
    return {"X-API-Key": k}


def _upload(client, key: str, gid: int, name: str = "讲义.docx") -> dict:
    """用 API Key 走服务化入库（同时验证上传端点）。"""
    return client.post(
        f"/v1/kb/groups/{gid}/documents",
        files={"file": (name, _docx_bytes([DOC_TEXT]))},
        headers=_key(key),
    )


# ===== 鉴权 =====


def test_v1_requires_valid_key(env):
    """缺失/伪造/吊销三种情况统一 401（不区分原因，防枚举）。"""
    c = env["client"]
    assert c.get("/v1/kb/groups").status_code == 401  # 无 Key
    assert c.get("/v1/kb/groups", headers=_key("zy_forged")).status_code == 401

    # 吊销后立刻失效
    tok = {"Authorization": f"Bearer {env['token_a']}"}
    kid = c.post("/api-keys", json={"name": "临时"}, headers=tok).json()["id"]
    temp_key = c.post("/api-keys", json={"name": "临时2"}, headers=tok).json()["key"]
    assert c.get("/v1/kb/groups", headers=_key(temp_key)).status_code == 200
    assert c.delete(f"/api-keys/{kid}", headers=tok).status_code == 200
    # 注意：上面吊销的是「临时」，temp_key 属于「临时2」——两把都吊销后再验
    items = c.get("/api-keys", headers=tok).json()
    for item in items:
        c.delete(f"/api-keys/{item['id']}", headers=tok)
    assert c.get("/v1/kb/groups", headers=_key(temp_key)).status_code == 401


def test_v1_account_isolation(env):
    """A 的 Key 只见 A 的分组；拿 B 的 group_id 检索/上传一律 404。"""
    c = env["client"]
    groups = c.get("/v1/kb/groups", headers=_key(env["key_a"])).json()
    assert [g["name"] for g in groups] == ["A组"]

    r = c.post(
        "/v1/rag/retrieve",
        json={"query": "学习率", "group_ids": [env["group_b"].id]},
        headers=_key(env["key_a"]),
    )
    assert r.status_code == 404

    assert _upload(c, env["key_a"], env["group_b"].id).status_code == 404


# ===== 能力契约 =====


def test_v1_retrieve_and_ask_roundtrip(env, monkeypatch):
    """上传 → 纯检索有命中 → 问答返回四字段（打桩生成，断言走真检索路径）。"""
    c = env["client"]
    up = _upload(c, env["key_a"], env["group_a"].id)
    assert up.status_code == 200 and up.json()["succeeded"] == 1

    r = c.post(
        "/v1/rag/retrieve",
        json={"query": "学习率过大会怎样", "group_ids": [env["group_a"].id]},
        headers=_key(env["key_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    hit = body["hits"][0]
    assert hit["file_name"] == "讲义.docx" and hit["page_no"] >= 1
    assert hit["text"] and "score" in hit and "group_id" in hit

    answer_stub = _StubGateway("学习率过大会震荡。")
    monkeypatch.setattr("app.agent.tools.gateway", answer_stub)
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway("qa"))

    r = c.post(
        "/v1/rag/ask",
        json={"question": "学习率过大会怎样", "group_ids": [env["group_a"].id]},
        headers=_key(env["key_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hit"] is True and body["intent"] == "qa"
    assert "来源" in body["answer"] and body["sources"][0]["file_name"] == "讲义.docx"
    assert answer_stub.calls == 1


def test_v1_fallback_never_calls_llm(env, monkeypatch):
    """空检索兜底：与内部接口同一条防幻觉硬闸门（v1 不是绕过闸门的后门）。"""
    from app.rag.prompts import FALLBACK_MESSAGE

    class _NoCall(_StubGateway):
        async def generate(self, **kwargs):
            raise AssertionError("v1 空召回分支严禁调用 LLM")

    for ns in ("app.agent.tools", "app.agent.intent", "app.memory.short_term"):
        monkeypatch.setattr(f"{ns}.gateway", _NoCall(""))

    r = env["client"].post(
        "/v1/rag/ask",
        json={"question": "光合作用的暗反应发生在哪里"},
        headers=_key(env["key_a"]),
    )
    body = r.json()
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE and body["sources"] == []


def test_v1_ask_stream_done_event(env, monkeypatch):
    """SSE 契约：流内以 done 事件收尾（无会话字段），与 /chat/ask 流式同构。"""
    c = env["client"]
    _upload(c, env["key_a"], env["group_a"].id)
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("学习率过大会震荡。"))
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway("qa"))

    r = c.post(
        "/v1/rag/ask/stream",
        json={"question": "学习率过大会怎样", "group_ids": [env["group_a"].id]},
        headers=_key(env["key_a"]),
    )
    assert r.status_code == 200
    events = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["t"] == "done"
    assert "answer" in events[-1] and "sources" in events[-1]


# ===== 背压（429）=====


class _BusyGate:
    """永远拿不到的闸门：模拟「前一个请求还在生成」的排队现场。"""

    async def acquire(self):
        import asyncio

        await asyncio.sleep(5)

    def release(self):
        pass


def test_v1_backpressure_returns_429(env, monkeypatch):
    """排队超过 V1_QUEUE_TIMEOUT → 429 + Retry-After（不无限挂起调用方）。"""
    monkeypatch.setattr("app.api.v1._v1_gate", _BusyGate())
    monkeypatch.setattr(settings, "v1_queue_timeout", 0.01)

    r = env["client"].post(
        "/v1/rag/retrieve",
        json={"query": "学习率"},
        headers=_key(env["key_a"]),
    )
    assert r.status_code == 429
    assert r.json()["error"]["code"] == 429
    assert r.headers.get("retry-after") == "5"
