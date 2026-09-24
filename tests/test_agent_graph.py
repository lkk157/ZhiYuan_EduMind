# -*- coding: utf-8 -*-
"""
M4 单测：LangGraph 路由——计算旁路/空召回零调用/意图分发/改写触发条件。
"""
import json

import pytest

from app.agent.graph import run_agent
from app.rag.retriever import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(text="梯度下降……", file_name="讲义.docx", page_no=1, score=0.9, group_id=1)


class _StubGateway:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return self.reply


class _NoCall:
    async def generate(self, **kwargs):
        raise AssertionError("该路径严禁调用 LLM")


def _patch_all(monkeypatch, *, retrieve_chunks, tools_reply, intent_reply="qa",
               rewrite_reply="", fail_on_llm=False):
    """一站式缝位：graph.retrieve + tools/intent/short_term 三个 gateway。

    fail_on_llm=True 时三个 gateway 全换成一调用即失败桩（空召回零调用硬校验）。
    """
    async def fake_retrieve(question, **kwargs):
        return retrieve_chunks

    monkeypatch.setattr("app.agent.graph.retrieve", fake_retrieve)
    if fail_on_llm:
        for ns in ("app.agent.tools", "app.agent.intent", "app.memory.short_term"):
            monkeypatch.setattr(f"{ns}.gateway", _NoCall())
        return
    tools_stub = _StubGateway(tools_reply)
    monkeypatch.setattr("app.agent.tools.gateway", tools_stub)
    monkeypatch.setattr("app.agent.intent.gateway", _StubGateway(intent_reply))
    rewrite_stub = _StubGateway(rewrite_reply)
    monkeypatch.setattr("app.memory.short_term.gateway", rewrite_stub)
    return tools_stub


@pytest.mark.asyncio
async def test_calc_bypass_skips_retrieve(monkeypatch):
    """纯算式在入口被正则接走：不检索、不分类，直送 calc 工具（结果确定性正确）。"""
    async def retrieve_should_not_be_called(question, **kwargs):
        raise AssertionError("计算旁路严禁触碰检索")

    monkeypatch.setattr("app.agent.graph.retrieve", retrieve_should_not_be_called)
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("先乘后加。"))

    out = await run_agent(question="23*47+108", user_id=1, group_ids=[1], history=[])
    assert out["intent"] == "calc"
    assert out["hit"] is True
    assert "1189" in out["answer"]


@pytest.mark.asyncio
async def test_empty_retrieval_falls_back_zero_llm(monkeypatch):
    """★防幻觉硬闸门（M4 布局版）：空召回 → 兜底，改写/分类/生成三缝全不许被碰。"""
    _patch_all(monkeypatch, retrieve_chunks=[], tools_reply="", fail_on_llm=True)

    out = await run_agent(question="任意问题", user_id=1, group_ids=[], history=[])
    assert out["hit"] is False
    assert "未找到" in out["answer"]
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_classify_routes_to_quiz(monkeypatch):
    """检索命中 + 分类=quiz → 出题工具执行（路由正确落到 quiz 节点）。"""
    quiz_json = json.dumps(
        {"type": "quiz", "questions": [{"type": "choice", "question": "Q", "options": ["A", "B"], "answer": "A"}]},
        ensure_ascii=False,
    )
    _patch_all(
        monkeypatch,
        retrieve_chunks=[_chunk()],
        tools_reply=quiz_json,
        intent_reply="quiz",
    )
    out = await run_agent(question="出一道题", user_id=1, group_ids=[1], history=[])
    assert out["intent"] == "quiz"
    assert out["hit"] is True
    assert json.loads(out["answer"])["type"] == "quiz"


@pytest.mark.asyncio
async def test_classify_routes_to_qa(monkeypatch):
    """检索命中 + 分类=qa → 答疑工具执行（sources/命中徽章口径与 M2 一致）。"""
    _patch_all(
        monkeypatch,
        retrieve_chunks=[_chunk()],
        tools_reply="学习率过大会震荡。【来源：x.pdf，第1页】",
        intent_reply="qa",
    )
    out = await run_agent(question="学习率过大会怎样", user_id=1, group_ids=[1], history=[])
    assert out["intent"] == "qa"
    assert out["hit"] is True
    assert out["sources"][0]["file_name"] == "讲义.docx"


@pytest.mark.asyncio
async def test_rewrite_called_only_with_history(monkeypatch):
    """改写触发条件：无历史零调用；有历史恰好一次（省推理 + 指代消解各得其所）。"""
    # 无历史
    stubs = _patch_all(
        monkeypatch, retrieve_chunks=[_chunk()], tools_reply="答案。", rewrite_reply="改写后"
    )
    # tools_stub 不是 rewrite 桩——单独拿 rewrite 桩
    await run_agent(question="它呢", user_id=1, group_ids=[1], history=[])
    # _patch_all 里 rewrite_stub 变量没返回，这里直接从命名空间取回桩对象断言
    from app.memory import short_term

    assert short_term.gateway.calls == 0

    # 有历史
    _patch_all(monkeypatch, retrieve_chunks=[_chunk()], tools_reply="答案。", rewrite_reply="改写后的问题")
    await run_agent(
        question="它呢",
        user_id=1,
        group_ids=[1],
        history=[{"role": "user", "content": "什么是梯度下降"}],
    )
    assert short_term.gateway.calls == 1
