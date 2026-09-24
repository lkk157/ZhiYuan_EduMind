# -*- coding: utf-8 -*-
"""
接口层单测（会话历史持久化）：落库/回看/越权/级联/兼容性的行为契约。

锁住的五条契约：
1. 传 conversation_id 提问 → 该会话能查到一问一答两条（顺序、hit、sources 还原）；
2. 不传 conversation_id → 会话列表不变（兼容性硬断言：demo_e2e/旧调用零影响）；
3. 兜底问答（hit=false）同样落库——历史不许选择性失忆；
4. 跨用户访问会话/消息一律 404（与分组/文档同款防探测纪律）；
5. 删会话 → 消息级联清空（手工级联在 SQLite 外键失效时也必须成立）。

测试姿势与 test_kb_chat 同款：裸 TestClient（不触发 startup 连真 MySQL）+
dependency_overrides 注入 SQLite + 内存向量库 + 假向量 + 假 LLM 网关。
"""
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.db import crud
from app.db.session import get_db
from app.rag import embeddings
from app.rag.prompts import FALLBACK_MESSAGE
from app.rag.vector_store import ChromaStore

DOC_TEXT = "梯度下降的学习率过大会导致损失函数震荡不收敛。"


def _docx_bytes(paras: list[str]) -> bytes:
    """python-docx 现造 docx 字节流（与 test_kb_chat 同款，单测不落盘临时文件）。"""
    from docx import Document

    import io

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
    """假 LLM 网关：返回固定文本并计数调用。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return self.reply


@pytest.fixture()
def env(db_session, monkeypatch, chroma_client, fake_embed_fn):
    """注入 SQLite 会话 + 内存向量库/假向量，造好 A、B 两个用户。"""
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


def _make_group_with_doc(client, token: str) -> int:
    """建分组 + 传一份 docx（命中问答的前置数据），返回 group_id。"""
    gid = client.post("/kb/groups", json={"name": "课件"}, headers=_auth(token)).json()["id"]
    r = client.post(
        f"/kb/groups/{gid}/documents",
        files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))},
        headers=_auth(token),
    )
    assert r.json()["succeeded"] == 1
    return gid


# ===== 核心：落库与回看 =====


def test_ask_persists_pair_and_reads_back(env, monkeypatch):
    """传 conversation_id：一问一答落库；回看顺序正确、sources 还原成结构化列表。"""
    c = env["client"]
    conv = c.post("/chat/conversations", json={"title": "学习率问题"}, headers=_auth(env["token_a"]))
    assert conv.status_code == 201
    cid = conv.json()["id"]

    gid = _make_group_with_doc(c, env["token_a"])
    monkeypatch.setattr("app.api.chat.gateway", _StubGateway("学习率过大会震荡。"))

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid], "conversation_id": cid},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hit"] is True
    assert body["conversation_id"] == cid  # 原样回显，前端据此确认落到了哪个会话

    msgs = c.get(f"/chat/conversations/{cid}/messages", headers=_auth(env["token_a"])).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "学习率过大会怎样"
    assert msgs[0]["hit"] is None  # 用户行无命中语义（NULL，防读侧误判成兜底）
    assert msgs[1]["hit"] is True
    # sources 从 JSON 字符串还原成结构化列表，且携带真实溯源字段
    assert msgs[1]["sources"][0]["file_name"] == "讲义.docx"
    assert msgs[1]["sources"][0]["page_no"] == 1


def test_ask_without_conversation_id_does_not_persist(env, monkeypatch):
    """★ 兼容性硬断言：不传 conversation_id → 会话列表保持为空（旧调用零副作用）。"""
    c = env["client"]
    gid = _make_group_with_doc(c, env["token_a"])
    monkeypatch.setattr("app.api.chat.gateway", _StubGateway("答。"))

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    assert r.json()["conversation_id"] is None
    assert c.get("/chat/conversations", headers=_auth(env["token_a"])).json() == []


def test_fallback_answer_also_persisted(env, monkeypatch):
    """兜底问答（hit=false）同样落库：回看必须能看到「这轮没命中」。"""
    c = env["client"]
    cid = c.post("/chat/conversations", json={"title": "闲聊"}, headers=_auth(env["token_a"])).json()["id"]

    async def fake_retrieve(question, **kwargs):
        return []

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)

    r = c.post(
        "/chat/ask",
        json={"question": "番茄炒蛋放多少盐", "conversation_id": cid},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    assert r.json()["hit"] is False

    msgs = c.get(f"/chat/conversations/{cid}/messages", headers=_auth(env["token_a"])).json()
    assert len(msgs) == 2
    assert msgs[1]["hit"] is False
    assert msgs[1]["content"] == FALLBACK_MESSAGE
    assert msgs[1]["sources"] == []


# ===== 会话列表与删除 =====


def test_conversation_list_count_and_recent_first(env, monkeypatch):
    """列表带 message_count、按最近活跃倒序（问答会把该会话顶到最前）。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    cid_old = c.post("/chat/conversations", json={"title": "早的"}, headers=tok).json()["id"]
    cid_new = c.post("/chat/conversations", json={"title": "晚的"}, headers=tok).json()["id"]
    assert cid_old != cid_new

    # 默认倒序=建得晚的在前
    convs = c.get("/chat/conversations", headers=tok).json()
    assert [x["id"] for x in convs] == [cid_new, cid_old]
    assert all(x["message_count"] == 0 for x in convs)

    # 给「早的」来一轮问答 → updated_at 刷新 → 它应顶到最前且消息数=2
    gid = _make_group_with_doc(c, env["token_a"])
    monkeypatch.setattr("app.api.chat.gateway", _StubGateway("答。"))
    c.post(
        "/chat/ask",
        json={"question": "学习率", "group_ids": [gid], "conversation_id": cid_old},
        headers=tok,
    )

    convs = c.get("/chat/conversations", headers=tok).json()
    assert [x["id"] for x in convs] == [cid_old, cid_new]
    by_id = {x["id"]: x for x in convs}
    assert by_id[cid_old]["message_count"] == 2
    assert by_id[cid_new]["message_count"] == 0


def test_delete_conversation_cascades_messages(env):
    """删会话 → 消息级联清空（手工级联，SQLite 外键失效下也必须成立）；再删 404。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    cid = c.post("/chat/conversations", json={"title": "待删"}, headers=tok).json()["id"]
    # 直接经 crud 塞一条消息（不走问答：本用例只测级联，不需要检索/LLM）
    conv = crud.get_conversation(env["db"], user_id=env["user_a"].id, conversation_id=cid)
    crud.append_message_pair(
        env["db"],
        conversation=conv,
        question="q",
        answer="a",
        hit=False,
        sources_json="[]",
    )
    assert c.get(f"/chat/conversations/{cid}/messages", headers=tok).json() != []

    assert c.delete(f"/chat/conversations/{cid}", headers=tok).status_code == 200
    # 会话没了 → 消息端点 404（防探测）；列表也回到空
    assert c.get(f"/chat/conversations/{cid}/messages", headers=tok).status_code == 404
    assert c.delete(f"/chat/conversations/{cid}", headers=tok).status_code == 404
    assert c.get("/chat/conversations", headers=tok).json() == []
    # 消息行物理删除（不只是外键悬空）：直接查表计数
    from sqlalchemy import func, select

    from app.db.models import Message

    count = env["db"].execute(select(func.count()).select_from(Message)).scalar()
    assert count == 0


# ===== 越权防线 =====


def test_cross_user_conversation_is_404(env):
    """B 访问 A 的会话（读消息/删除）一律 404；借 A 的会话提问也 404。"""
    c = env["client"]
    cid = c.post(
        "/chat/conversations", json={"title": "A的会话"}, headers=_auth(env["token_a"])
    ).json()["id"]

    assert c.get(f"/chat/conversations/{cid}/messages", headers=_auth(env["token_b"])).status_code == 404
    assert c.delete(f"/chat/conversations/{cid}", headers=_auth(env["token_b"])).status_code == 404

    r = c.post(
        "/chat/ask",
        json={"question": "越权提问", "conversation_id": cid},
        headers=_auth(env["token_b"]),
    )
    assert r.status_code == 404
    assert "会话不存在" in r.json()["error"]["message"]

    # 对照：A 自己访问正常
    assert c.get(f"/chat/conversations/{cid}/messages", headers=_auth(env["token_a"])).status_code == 200


def test_empty_title_falls_back_to_default(env):
    """空标题兜底「新对话」+ 截断到 128——服务端复校，不信任客户端。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    r = c.post("/chat/conversations", json={"title": "   "}, headers=tok)
    assert r.status_code == 201
    assert r.json()["title"] == "新对话"

    r = c.post("/chat/conversations", json={"title": "长" * 300}, headers=tok)
    assert len(r.json()["title"]) == 128
