# -*- coding: utf-8 -*-
"""
M4 单测：短期记忆——滑窗截取、无历史零调用、改写失败回退原问题。
"""
import pytest

from app.core.config import settings
from app.db import crud
from app.memory import short_term


class _StubGateway:
    """假网关：可返回指定文本或计数（断言「有没有被调」）。"""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return self.reply


def _make_conv_with_messages(db, user, n_messages: int) -> int:
    """造一个含 n_messages 条消息的会话，返回 conversation_id。

    append_message_pair 一次写两条（问+答），所以按「轮数 = 条数 // 2」循环——
    造数粒度按条数与滑窗断言的单位保持一致（第一版按轮循环多造了一倍，断言即错）。
    """
    conv = crud.create_conversation(db, user_id=user.id, title="t")
    for i in range(n_messages // 2):
        crud.append_message_pair(
            db,
            conversation=conv,
            question=f"问{i}",
            answer=f"答{i}",
            hit=True,
            sources_json="[]",
        )
    return conv.id


def test_load_window_slices_last_n(db_session, monkeypatch):
    """滑窗只取最近 N 条（配置 short_term_window），且时间正序。"""
    user = crud.create_user(db_session, username="u1", password_hash="x")
    cid = _make_conv_with_messages(db_session, user, 8)  # 8 条
    monkeypatch.setattr(settings, "short_term_window", 4)

    window = short_term.load_window(db_session, user_id=user.id, conversation_id=cid)
    assert len(window) == 4
    # 最近 4 条 = 第 3 轮问答（问2/答2/问3/答3）——头两轮被挤出窗口
    contents = [m["content"] for m in window]
    assert contents[0] == "问2"
    assert contents[-1] == "答3"
    assert all(m["role"] in {"user", "assistant"} for m in window)


def test_load_window_none_conversation_is_empty(db_session):
    """conversation_id=None（无状态问答）→ 空历史，后续改写零调用。"""
    assert short_term.load_window(db_session, user_id=1, conversation_id=None) == []


@pytest.mark.asyncio
async def test_rewrite_skips_without_history(monkeypatch):
    """★无历史不触发改写——零模型调用（省一次推理 + 保证空召回路径的零调用属性）。"""
    stub = _StubGateway("不该被调")
    monkeypatch.setattr("app.memory.short_term.gateway", stub)

    out = await short_term.rewrite_question("梯度下降是什么", [])
    assert out == "梯度下降是什么"
    assert stub.calls == 0


@pytest.mark.asyncio
async def test_rewrite_uses_history_and_returns_llm_output(monkeypatch):
    """有历史 → 调一次 LLM，返回其输出（独立完整问句）。"""
    stub = _StubGateway("  梯度下降的学习率过大会怎样  ")
    monkeypatch.setattr("app.memory.short_term.gateway", stub)

    history = [
        {"role": "user", "content": "什么是梯度下降"},
        {"role": "assistant", "content": "一种优化方法……"},
    ]
    out = await short_term.rewrite_question("它呢？", history)
    assert stub.calls == 1
    assert out == "梯度下降的学习率过大会怎样"  # 已 strip


@pytest.mark.asyncio
async def test_rewrite_empty_output_falls_back_to_original(monkeypatch):
    """LLM 返回空串 → 回退原问题（改写是增强，绝不能让提问失败）。"""
    monkeypatch.setattr("app.memory.short_term.gateway", _StubGateway("   "))
    out = await short_term.rewrite_question("原问题", [{"role": "user", "content": "前文"}])
    assert out == "原问题"
