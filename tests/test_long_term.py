# -*- coding: utf-8 -*-
"""
M5 单测：长效记忆——快路零开销、语义召回降级、双写顺序、周报提炼与解析。
"""
import pytest

from app.core.config import settings
from app.core.exceptions import AppError
from app.db import crud
from app.memory import long_term


class _StubGateway:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.kwargs: dict = {}

    async def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.reply


class _BoomEmbed:
    """embedding 一被调用即失败——用于证明快路/降级路径零依赖。"""

    async def __call__(self, texts):
        raise AssertionError("该路径严禁调用 embedding")


class _RecMemoryStore:
    """记录 upsert 的假记忆向量库；query 可预置返回。"""

    def __init__(self, query_hits=None):
        self.upserts = []
        self.query_hits = query_hits or []

    async def upsert(self, *, ids, texts, metadatas):
        self.upserts.append((list(ids), list(texts)))

    def query(self, vector, top_k):
        return self.query_hits[:top_k]


def test_make_weak_fact_template():
    """错题事实零 LLM：模板拼句、截断题干、带教材锚。"""
    fact = long_term.make_weak_fact("什么是" + "长" * 100, "2.3.1_单链表的定义.pdf")
    assert fact.startswith("学生答错过：")
    assert "（教材出处：2.3.1_单链表的定义.pdf）" in fact
    assert len(fact) < 200  # 题干截断到 60 字


@pytest.mark.asyncio
async def test_recall_fast_path_no_facts_zero_cost(db_session, monkeypatch):
    """无记忆快路：返回 [] 且严禁碰 embedding（新用户每次提问零额外开销）。"""
    user = crud.create_user(db_session, username="fresh", password_hash="x")
    monkeypatch.setattr("app.rag.embeddings.embed_texts", _BoomEmbed())
    out = await long_term.recall_memories(db_session, user_id=user.id, question="任意")
    assert out == []


@pytest.mark.asyncio
async def test_recall_semantic_failure_degrades_to_recency(db_session, monkeypatch):
    """语义段挂掉 → 静默降级「最近条」——注入是增强，绝不让提问失败。"""
    user = crud.create_user(db_session, username="u2", password_hash="x")
    crud.create_memory_fact(db_session, user_id=user.id, kind=long_term.KIND_WEAK, content="薄弱事实A")
    crud.create_memory_fact(db_session, user_id=user.id, kind=long_term.KIND_WEAK, content="薄弱事实B")

    monkeypatch.setattr("app.rag.embeddings.embed_texts", _BoomEmbed())
    out = await long_term.recall_memories(db_session, user_id=user.id, question="任意")
    assert out  # 降级成功
    assert out[0] == "薄弱事实B"  # 最近条在前（id desc）


@pytest.mark.asyncio
async def test_recall_merges_semantic_and_recency_capped(db_session, monkeypatch):
    """语义+最近合并去重、按 memory_top_k 封顶、单条截断 150 字。"""
    user = crud.create_user(db_session, username="u3", password_hash="x")
    long_content = "薄弱" * 100  # 200+ 字
    crud.create_memory_fact(db_session, user_id=user.id, kind=long_term.KIND_WEAK, content=long_content)
    crud.create_memory_fact(db_session, user_id=user.id, kind=long_term.KIND_INSIGHT, content="洞察X")

    class _Hit:
        def __init__(self, text):
            self.text = text

    mem = _RecMemoryStore(query_hits=[_Hit("语义命中Y"), _Hit(long_content)])  # 第二条与最近条重复
    monkeypatch.setattr("app.memory.long_term.memory_store_for", lambda uid: mem)
    monkeypatch.setattr("app.rag.embeddings.embed_texts", lambda texts: _Await([[0.1, 0.2]]))

    out = await long_term.recall_memories(db_session, user_id=user.id, question="任意")
    # 语义命中Y + 洞察X + 薄弱A（去重后），全部 ≤ memory_top_k 条、单条 ≤150 字
    assert out[0] == "语义命中Y"
    assert len(out) <= settings.memory_top_k
    assert all(len(x) <= 150 for x in out)


class _Await(list):
    def __await__(self):
        yield from ()
        return list(self)


@pytest.mark.asyncio
async def test_write_fact_mysql_first_index_best_effort(db_session, monkeypatch):
    """双写顺序：索引写挂了，事实照样落 MySQL（先事实后索引，绝不因索引丢记忆）。"""
    user = crud.create_user(db_session, username="u4", password_hash="x")

    class _BadStore:
        async def upsert(self, **kwargs):
            raise RuntimeError("chroma 挂了")

    monkeypatch.setattr("app.memory.long_term.memory_store_for", lambda uid: _BadStore())
    fact = await long_term.write_fact(
        db_session, user_id=user.id, kind=long_term.KIND_WEAK, content="断网也能记住"
    )
    rows = crud.list_memory_facts(db_session, user_id=user.id)
    assert [f.content for f in rows] == ["断网也能记住"]
    assert fact.id


def test_parse_report_variants():
    """周报 JSON 解析：裸 JSON / 围栏包裹 / 缺字段 → None（不给半截报告）。"""
    ok = long_term.parse_report('{"report": "# 周报", "weak_points": ["薄弱1"]}')
    assert ok == {"report": "# 周报", "weak_points": ["薄弱1"]}

    fenced = long_term.parse_report('```json\n{"report": "R", "weak_points": ["a"]}\n```')
    assert fenced is not None

    assert long_term.parse_report("没有 JSON") is None
    assert long_term.parse_report('{"report": "", "weak_points": []}') is None
    assert long_term.parse_report('{"report": "R", "weak_points": "不是数组"}') is None


@pytest.mark.asyncio
async def test_generate_report_end_to_end(db_session, monkeypatch):
    """周报生成：素材聚合 → LLM 提炼 → 报告与薄弱点双写入库；空素材 AppError 400。"""
    user = crud.create_user(db_session, username="u5", password_hash="x")
    conv = crud.create_conversation(db_session, user_id=user.id, title="t")
    crud.append_message_pair(
        db_session, conversation=conv, question="q", answer="a", hit=True, sources_json="[]"
    )

    mem = _RecMemoryStore()
    monkeypatch.setattr("app.memory.long_term.memory_store_for", lambda uid: mem)
    stub = _StubGateway('{"report": "# 本周表现\\n不错", "weak_points": ["单链表插入不熟"]}')
    monkeypatch.setattr("app.memory.long_term.gateway", stub)

    out = await long_term.generate_report(db_session, user_id=user.id)
    assert stub.calls == 1
    assert out["report"].startswith("# 本周表现")
    assert out["weak_points"] == ["单链表插入不熟"]
    # 双写落点：报告 1 条 + 薄弱点 1 条；向量索引也各写了 1 条
    kinds = sorted(f.kind for f in crud.list_memory_facts(db_session, user_id=user.id))
    assert kinds == [long_term.KIND_REPORT, long_term.KIND_WEAK]
    assert len(mem.upserts) == 2

    # 空素材用户 → AppError 400（人话，不进 LLM）
    lonely = crud.create_user(db_session, username="lonely", password_hash="x")
    with pytest.raises(AppError):
        await long_term.generate_report(db_session, user_id=lonely.id)
    assert stub.calls == 1  # 没素材就不许调模型


def test_build_prompt_background_injection():
    """学情背景注入：qa 与出题 prompt 都带护栏措辞（禁复述/禁点破）。"""
    from app.rag.prompts import build_quiz_prompt, build_qa_prompt

    system_qa, _ = build_qa_prompt("q", [], background=["学生不熟单链表插入"])
    assert "学情背景" in system_qa and "禁止在回答中复述" in system_qa
    # 无背景 → 段落不存在（默认路径与既有测试零影响）
    system_plain, _ = build_qa_prompt("q", [])
    assert "学情背景" not in system_plain

    system_quiz, _ = build_quiz_prompt("q", [], background=["薄弱点X"])
    assert "学情背景" in system_quiz and "优先围绕" in system_quiz
    assert "提及学生的过往错误" in system_quiz
