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
    """假 LLM 网关：返回固定文本、计数调用、记录最近一次 kwargs（断言 prompt/温度用）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.kwargs: dict = {}

    async def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.reply


class _NoCallGateway(_StubGateway):
    """一旦被调用就让测试失败——专用于「严禁调 LLM」分支的硬校验。"""

    async def generate(self, **kwargs):
        raise AssertionError("防幻觉硬闸门失效：该分支严禁调用 LLM")


def _stub_agent(monkeypatch, reply: str, intent_label: str = "qa") -> _StubGateway:
    """M4 缝位：生成/步骤调用在 tools、意图分类在 intent——两处分别打桩。

    返回的是「生成桩」（tools.gateway）：断言 calls==1 指的是答案生成这一次，
    分类调用有自己的桩互不计数——这正是缝位拆分的意义。
    """
    answer_stub = _StubGateway(reply)
    monkeypatch.setattr("app.agent.tools.gateway", answer_stub)
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway(intent_label))
    return answer_stub


def _nocall_agent(monkeypatch) -> None:
    """三个可能触碰 LLM 的命名空间全部换成「一调用就失败」桩。

    M4 后空召回路径涉及 改写/分类/生成 三个潜在调用点——只堵一个缝的测试
    在缝位搬家后会静默失效（补丁打在没人调用的命名空间上，测试照过但没验到），
    三个一起堵才是完整硬闸门。
    """
    for ns in ("app.agent.tools", "app.agent.intent", "app.memory.short_term"):
        monkeypatch.setattr(f"{ns}.gateway", _NoCallGateway(""))


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
    """上传→增量计数；同名同内容重传 skipped_identical；改内容重传 changed>=1（批量契约）。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    data = _docx_bytes([DOC_TEXT])

    r = c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", data)}, headers=_auth(env["token_a"]))
    assert r.status_code == 200
    body = r.json()
    # 批量响应契约：{results:[...], succeeded, failed}（单文件 = results 一项）
    assert body["succeeded"] == 1 and body["failed"] == 0
    result = body["results"][0]
    assert result["ok"] is True
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
    assert r.json()["results"][0]["skipped_identical"] is True

    # 同名改内容重传：chunk 级增量（0 号块 hash 变了 → changed）
    data2 = _docx_bytes([DOC_TEXT + "补充：应使用学习率衰减。"])
    r = c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", data2)}, headers=_auth(env["token_a"]))
    result = r.json()["results"][0]
    assert result["skipped_identical"] is False
    assert result["changed"] + result["added"] >= 1


def test_batch_upload_rejects_too_many_files(env):
    """单批超过 MAX_UPLOAD_FILES（5）→ 整批 400，服务端复校（客户端限制不是安全边界）。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "限"}, headers=_auth(env["token_a"])).json()["id"]
    # 6 个文件（内容无关：批次校验先于任何落盘/解析）
    parts = [("file", (f"f{i}.docx", b"x", "application/octet-stream")) for i in range(6)]
    r = c.post(f"/kb/groups/{gid}/documents", files=parts, headers=_auth(env["token_a"]))
    assert r.status_code == 400
    assert "最多" in r.json()["error"]["message"]


def test_batch_upload_rejects_oversized_total(env, monkeypatch):
    """总大小超过上限 → 整批 400（用 max_total_mb=0 构造超限，避免真造 200MB 文件）。"""
    c = env["client"]
    monkeypatch.setattr(settings, "max_upload_total_mb", 0)
    gid = c.post("/kb/groups", json={"name": "限"}, headers=_auth(env["token_a"])).json()["id"]
    r = c.post(
        f"/kb/groups/{gid}/documents",
        files={"file": ("a.docx", _docx_bytes([DOC_TEXT]))},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 400
    assert "总大小" in r.json()["error"]["message"]


def test_batch_upload_continues_on_single_failure(env):
    """混批：坏文件（不支持的类型）只标自身失败，好文件照常入库——单文件失败不连坐。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "混批"}, headers=_auth(env["token_a"])).json()["id"]
    parts = [
        ("file", ("好.docx", _docx_bytes([DOC_TEXT]), "application/octet-stream")),
        ("file", ("坏.txt", b"plain text", "text/plain")),
    ]
    r = c.post(f"/kb/groups/{gid}/documents", files=parts, headers=_auth(env["token_a"]))
    assert r.status_code == 200
    body = r.json()
    assert body["succeeded"] == 1 and body["failed"] == 1
    by_name = {item["file_name"]: item for item in body["results"]}
    assert by_name["好.docx"]["ok"] is True
    assert by_name["坏.txt"]["ok"] is False
    assert "不支持" in by_name["坏.txt"]["error"]
    # 好文件确实进了文档列表
    docs = c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_a"])).json()
    assert [d["file_name"] for d in docs] == ["好.docx"]


# ===== 问答：命中 / 兜底 =====


def test_ask_hit_appends_real_sources_and_strips_fake(env, monkeypatch):
    """命中：强制拼接真实【来源】；模型自写的伪来源被删（三层防线合成验证）。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    stub = _stub_agent(monkeypatch, "学习率过大会震荡。【来源：编造.pdf，第99页】")

    r = c.post(
        "/chat/ask",
        json={"question": "梯度下降的学习率过大会导致什么", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hit"] is True
    assert body["intent"] == "qa"  # M4 意图路由标签随响应返回（前端徽章数据源）
    assert stub.calls == 1
    assert "编造" not in body["answer"]  # 伪来源被 sanitize 删掉
    assert "来源：" in body["answer"]  # 真来源被强制拼接
    assert "讲义.docx" in body["answer"]
    assert body["sources"][0]["file_name"] == "讲义.docx"
    assert body["sources"][0]["page_no"] == 1


def test_guide_mode_switches_system_prompt(env, monkeypatch):
    """引导式答疑开关：guide_mode=True 时 qa 的 system 含引导铁律，False 时不含。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    stub = _stub_agent(monkeypatch, "学习率过大会震荡。")
    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid], "guide_mode": True},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    assert "引导式答疑模式" in stub.kwargs.get("system", "")

    # 关闭（缺省）对照：同一批断言反向成立，证明开关真的切了 prompt 而不是恒定文案
    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    assert "引导式答疑模式" not in stub.kwargs.get("system", "")


def test_ask_fallback_never_calls_llm(env, monkeypatch):
    """★ 防幻觉硬闸门：空检索直接兜底，LLM 一旦被调用测试即失败。"""
    c = env["client"]

    async def fake_retrieve(question, **kwargs):
        return []

    monkeypatch.setattr("app.agent.graph.retrieve", fake_retrieve)
    _nocall_agent(monkeypatch)  # 改写/分类/生成三缝全堵——任一被调即测试失败

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
    _nocall_agent(monkeypatch)
    r = c.post(
        "/chat/ask",
        json={"question": "光合作用的暗反应发生在叶绿体基质吗", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    body = r.json()
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE


# ===== 资料不足闸门（模型不知道就不输出来源，2026-09-24 产品需求）=====


def test_insufficient_answer_falls_back_without_sources(env, monkeypatch):
    """模型答「根据现有资料无法回答…」→ 整体走兜底：固定话术、无【来源】、sources=[]。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    # 提问必须能命中检索（用与课件相关的问法）：本用例专测「检索到块、但模型判定
    # 资料不足」的第二道闸门——若用无关提问，会在空检索分支就被兜底、LLM 根本不会被调
    stub = _stub_agent(monkeypatch, "根据现有资料无法回答该知识点，资料中缺少相关章节。")

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert stub.calls == 1  # 答案生成确实被调了（本路径与空检索不同）
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE
    assert "来源" not in body["answer"]  # 未拼接【来源】
    assert body["sources"] == []


def test_insufficient_secondary_phrase_falls_back(env, monkeypatch):
    """副判据：模型没按固定句式、但首句含强特征短语（很抱歉，资料中未提及…）→ 同样兜底。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    _stub_agent(monkeypatch, "很抱歉，资料中未提及该内容。")

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    body = r.json()
    assert body["hit"] is False
    assert body["answer"] == FALLBACK_MESSAGE
    assert body["sources"] == []


def test_normal_answer_with_bududiao_still_gets_sources(env, monkeypatch):
    """★假阳性守卫：正文含「不知道」但有实质内容 → 照常命中、照常拼来源，绝不误杀。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    _stub_agent(monkeypatch, "很多同学不知道这个原理，正确做法是调小学习率或使用衰减策略。")

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid]},
        headers=_auth(env["token_a"]),
    )
    body = r.json()
    assert body["hit"] is True
    assert "【来源" in body["answer"]  # 真来源照常追加
    assert body["sources"]  # 出口来源列表非空


def test_insufficient_answer_persisted_as_fallback(env, monkeypatch):
    """资料不足的轮次按 hit=false 落库——历史回看显示兜底样式、无来源卡片。"""
    c = env["client"]
    cid = c.post("/chat/conversations", json={"title": "不足测试"}, headers=_auth(env["token_a"])).json()["id"]
    gid = c.post("/kb/groups", json={"name": "课件"}, headers=_auth(env["token_a"])).json()["id"]
    c.post(f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"]))

    _stub_agent(monkeypatch, "根据现有资料无法回答。")

    r = c.post(
        "/chat/ask",
        json={"question": "学习率过大会怎样", "group_ids": [gid], "conversation_id": cid},
        headers=_auth(env["token_a"]),
    )
    assert r.json()["hit"] is False

    msgs = c.get(f"/chat/conversations/{cid}/messages", headers=_auth(env["token_a"])).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["hit"] is False
    assert msgs[1]["content"] == FALLBACK_MESSAGE
    assert msgs[1]["sources"] == []


# ===== 越权防线 =====


def test_cross_user_access_is_404(env):
    """B 操作 A 的分组/文档一律 404（不泄漏存在性）；借 A 的分组提问也 404。"""
    c = env["client"]
    gid = c.post("/kb/groups", json={"name": "A的组"}, headers=_auth(env["token_a"])).json()["id"]
    did = c.post(
        f"/kb/groups/{gid}/documents", files={"file": ("讲义.docx", _docx_bytes([DOC_TEXT]))}, headers=_auth(env["token_a"])
    ).json()["results"][0]["doc_id"]

    r = c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_b"]))
    assert r.status_code == 404
    assert "error" in r.json()

    r = c.delete(f"/kb/documents/{did}", headers=_auth(env["token_b"]))
    assert r.status_code == 404

    r = c.post("/chat/ask", json={"question": "梯度下降", "group_ids": [gid]}, headers=_auth(env["token_b"]))
    assert r.status_code == 404

    # 对照：A 自己访问正常
    assert c.get(f"/kb/groups/{gid}/documents", headers=_auth(env["token_a"])).status_code == 200
