# -*- coding: utf-8 -*-
"""
体验增强包单测：SSE 流式问答事件契约 + 原页图片端点（含越权/越界）。

为什么 SSE 事件序必须测死：前端解析器只认 data 行里的 t 字段——
事件名/字段一漂移，前端就是黑屏或静默丢答案；且「done 才是净化后契约」
（闸门/伪来源净化发生在流完之后）必须由断言守住，否则流式会绕过防幻觉三层。
"""
import io
import json

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import UpstreamError
from app.core.security import create_access_token
from app.db import crud
from app.db.session import get_db
from app.rag import embeddings
from app.rag.prompts import FALLBACK_MESSAGE
from app.rag.vector_store import ChromaStore
from tests.test_kb_chat import _StubGateway, _docx_bytes, _patch_store_and_embed

DOC_TEXT = "梯度下降的学习率过大会导致损失函数震荡不收敛。"


@pytest.fixture()
def env(db_session, monkeypatch, chroma_client, fake_embed_fn):
    """SQLite + 内存向量库 + A/B 两用户（与 test_kb_chat 同款姿势）。"""
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


class _StreamGateway:
    """流式假网关：按片产出 token；流式分支一旦走了非流式方法即失败。"""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces
        self.calls = 0

    async def generate_stream(self, **kwargs):
        self.calls += 1
        for piece in self.pieces:
            yield piece

    async def generate(self, **kwargs):
        raise AssertionError("流式分支严禁退回非流式 generate（打字机效果会丢失）")


def _stub_stream(monkeypatch, pieces: list[str], intent_label: str = "qa") -> _StreamGateway:
    gateway = _StreamGateway(pieces)
    monkeypatch.setattr("app.agent.tools.gateway", gateway)
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway(intent_label))
    return gateway


def _post_ask_sse(client, payload: dict, headers: dict) -> list[dict]:
    """打 SSE 问答并解析 data 行 → 事件 dict 列表。"""
    events: list[dict] = []
    with client.stream("POST", "/chat/ask", json=payload, headers=headers) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def _upload_doc(client, token: str) -> int:
    gid = client.post("/kb/groups", json={"name": "课件"}, headers=_auth(token)).json()["id"]
    r = client.post(
        f"/kb/groups/{gid}/documents",
        files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))},
        headers=_auth(token),
    )
    assert r.json()["succeeded"] == 1
    return gid


# ===== SSE 流式问答 =====


def test_ask_stream_deltas_then_done_and_persisted(env, monkeypatch):
    """事件序：N 条 delta → 1 条 done（done=净化后契约）；done 后会话已落库。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    gid = _upload_doc(c, env["token_a"])
    cid = c.post("/chat/conversations", json={"title": "流式"}, headers=tok).json()["id"]
    stub = _stub_stream(monkeypatch, ["学习率", "过大会震荡。"])

    events = _post_ask_sse(
        c,
        {"question": "学习率过大会怎样", "group_ids": [gid], "conversation_id": cid, "stream": True},
        tok,
    )
    deltas = [e["v"] for e in events if e["t"] == "delta"]
    dones = [e for e in events if e["t"] == "done"]
    errors = [e for e in events if e["t"] == "error"]
    assert deltas == ["学习率", "过大会震荡。"]  # 逐 token 顺序完整
    assert not errors
    assert stub.calls == 1
    assert len(dones) == 1
    done = dones[0]
    assert done["conversation_id"] == cid
    assert done["hit"] is True and done["intent"] == "qa"
    assert "学习率" in done["answer"] and "【来源" in done["answer"]
    assert done["sources"][0]["file_name"] == "讲义.docx"

    # done 事件发出时服务端已落库（前端收到即可信）
    msgs = c.get(f"/chat/conversations/{cid}/messages", headers=tok).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == done["answer"]


def test_ask_stream_insufficient_gate_in_done(env, monkeypatch):
    """流式后处理闸门不变：流出去的是原始「无法回答」，done 必须是兜底口径。

    提问必须用能命中检索的问法（过短的词在假向量下会跌破门槛走空召回，
    那测的就是另一条路径了）——calls==1 断言证明本用例真的走到了生成+闸门。
    """
    c = env["client"]
    tok = _auth(env["token_a"])
    gid = _upload_doc(c, env["token_a"])
    stub = _stub_stream(monkeypatch, ["根据现有资料", "无法回答。"])

    events = _post_ask_sse(
        c, {"question": "学习率过大会怎样", "group_ids": [gid], "stream": True}, tok
    )
    done = next(e for e in events if e["t"] == "done")
    assert stub.calls == 1  # 确认走到流式生成（而不是空召回提前兜底）
    assert done["hit"] is False
    assert done["answer"] == FALLBACK_MESSAGE  # 闸门在流完后统一执行
    assert done["sources"] == []


def test_ask_stream_error_event_no_done(env, monkeypatch):
    """生成中断 → error 事件收尾、绝不补 done（前端据此走失败分支）。"""
    c = env["client"]
    tok = _auth(env["token_a"])
    gid = _upload_doc(c, env["token_a"])

    class _BoomGateway(_StreamGateway):
        async def generate_stream(self, **kwargs):
            raise UpstreamError("Ollama 挂了")
            yield  # noqa: W0101 —— 仅用于标记为 async generator

    monkeypatch.setattr("app.agent.tools.gateway", _BoomGateway([]))
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway("qa"))

    events = _post_ask_sse(
        c, {"question": "学习率过大会怎样", "group_ids": [gid], "stream": True}, tok
    )
    assert [e["t"] for e in events] == ["error"]  # 无 delta 无 done
    assert "Ollama 挂了" in events[0]["message"]


def test_ask_stream_false_returns_json(env, monkeypatch):
    """stream=false（缺省）保持既有 JSON 契约——老调用方零感知的硬回归。"""
    from tests.test_kb_chat import _stub_agent

    c = env["client"]
    tok = _auth(env["token_a"])
    gid = _upload_doc(c, env["token_a"])
    _stub_agent(monkeypatch, "学习率过大会震荡。")

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=tok,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hit"] is True and "answer" in body and "conversation_id" in body


# ===== 原页图片端点 =====


def _make_pdf(path, pages: int = 3):
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_page_image_pdf_and_guards(env, tmp_path):
    """PDF 渲染 200 出 PNG；越界页码/非本人/不支持格式 一律人话 404。"""
    c = env["client"]
    tok_a = _auth(env["token_a"])
    gid = c.post("/kb/groups", json={"name": "预览"}, headers=tok_a).json()["id"]

    pdf_path = tmp_path / "课件.pdf"
    _make_pdf(pdf_path, pages=3)
    crud.create_document(
        env["db"],
        user_id=env["user_a"].id,
        group_id=gid,
        file_name="课件.pdf",
        file_path=pdf_path,
        file_hash="0" * 64,
    )

    # 正常渲染：PNG 魔数 + 第 2 页
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "课件.pdf", "page_no": 2},
        headers=tok_a,
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    # 页码越界
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "课件.pdf", "page_no": 99},
        headers=tok_a,
    )
    assert r.status_code == 404
    assert "超出" in r.json()["error"]["message"]

    # 非本人 → 404（防探测，不泄漏文档存在性）
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "课件.pdf", "page_no": 1},
        headers=_auth(env["token_b"]),
    )
    assert r.status_code == 404

    # Word 不支持 → 人话 404（宁可不给，不给错页）
    docx_path = tmp_path / "讲义.docx"
    docx_path.write_bytes(b"stub")
    crud.create_document(
        env["db"],
        user_id=env["user_a"].id,
        group_id=gid,
        file_name="讲义.docx",
        file_path=docx_path,
        file_hash="1" * 64,
    )
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "讲义.docx", "page_no": 1},
        headers=tok_a,
    )
    assert r.status_code == 404
    assert "暂不支持" in r.json()["error"]["message"]


def test_page_image_plain_png_bytes_passthrough(env, tmp_path):
    """图片原件直出（页码=1）：字节原样返回、media type 正确。"""
    c = env["client"]
    tok_a = _auth(env["token_a"])
    gid = c.post("/kb/groups", json={"name": "预览"}, headers=tok_a).json()["id"]
    png_path = tmp_path / "图.png"
    payload = b"\x89PNG\r\n\x1a\nfake-payload"
    png_path.write_bytes(payload)
    crud.create_document(
        env["db"],
        user_id=env["user_a"].id,
        group_id=gid,
        file_name="图.png",
        file_path=png_path,
        file_hash="2" * 64,
    )
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "图.png", "page_no": 1},
        headers=tok_a,
    )
    assert r.status_code == 200
    assert r.content == payload
    # 图片只有 1 页，页码 2 → 越界
    r = c.get(
        f"/kb/groups/{gid}/page-image",
        params={"file_name": "图.png", "page_no": 2},
        headers=tok_a,
    )
    assert r.status_code == 404


# ===== 工具层流式回调 =====


@pytest.mark.asyncio
async def test_qa_tool_on_delta_forwards_and_postprocesses(monkeypatch):
    """on_delta 收到每一片；最终返回仍是净化+闸门+拼来源后的结果。"""
    from app.agent import tools
    from app.rag.retriever import RetrievedChunk

    monkeypatch.setattr(
        "app.agent.tools.gateway",
        _StreamGateway(["学习率过大", "会震荡。【来源：假.pdf，第9页】"]),
    )
    got: list[str] = []
    chunk = RetrievedChunk(text=DOC_TEXT, file_name="讲义.docx", page_no=1, score=0.9, group_id=1)
    out = await tools.qa_tool("学习率", [chunk], on_delta=got.append)
    assert got == ["学习率过大", "会震荡。【来源：假.pdf，第9页】"]  # 原始 token 逐片转发
    assert "假.pdf" not in out["answer"]  # 伪来源仍在流完后被净化
    assert "【来源" in out["answer"] and out["hit"] is True
