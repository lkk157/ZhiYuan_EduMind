# -*- coding: utf-8 -*-
"""
M4 单测：意图分类——解析容错、非法默认 qa、温度/输出受控（风险 R3 的验收落点）。


"""
import pytest

from app.agent.intent import (
    INTENT_QA,
    INTENT_QUIZ,
    VALID_INTENTS,
    classify_intent,
    parse_intent,
)


class _CaptureGateway:
    """记录 kwargs 并返回固定文本（断言 temperature/num_predict 用）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.kwargs: dict = {}

    async def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("qa", "qa"),
        ("quiz", "quiz"),
        ("SUMMARY", "summary"),  # 大小写不敏感
        ("  calc  ", "calc"),  # 去空白
        ("quiz（出题）", "quiz"),  # 模型带尾巴输出
        ("意图是 summary 。", "summary"),  # 模型啰嗦输出
        ("", INTENT_QA),  # 空输出
        ("我不知道", INTENT_QA),  # 非法输出 → 默认答疑（路由永不卡死）
        ("qa; DROP TABLE", "qa"),  # 垃圾里含合法标签 → 宽松命中 qa（本来就该 qa）
    ],
)
def test_parse_intent_tolerant(raw, expected):
    """解析容错矩阵：精确/大小写/带尾巴/空/非法全部落在合法标签内。"""
    label = parse_intent(raw)
    assert label in VALID_INTENTS
    assert label == expected


@pytest.mark.asyncio
async def test_classify_intent_low_temp_small_output(monkeypatch):
    """分类调用必须 temperature=0 且输出≤16 token（确定性任务 + 红线：输出受控）。"""
    stub = _CaptureGateway("quiz")
    monkeypatch.setattr("app.agent.intent.gateway", stub)

    label = await classify_intent("出5道关于导数的题")
    assert label == INTENT_QUIZ
    assert stub.kwargs["temperature"] == 0.0
    assert stub.kwargs["num_predict"] <= 16


@pytest.mark.asyncio
async def test_classify_intent_garbage_falls_back_qa(monkeypatch):
    """模型抽风输出乱码 → parse 兜底成 qa（端到端默认路由）。"""
    monkeypatch.setattr("app.agent.intent.gateway", _CaptureGateway("（模型今天不想分类）"))
    assert await classify_intent("任意问题") == INTENT_QA
