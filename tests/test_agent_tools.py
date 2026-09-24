# -*- coding: utf-8 -*-
"""
M4 单测：四个工具——安全求值（含恶意拒绝）、试题解析、资料不足闸门、判分。
"""
import json

import pytest

from app.agent import tools
from app.rag.prompts import FALLBACK_MESSAGE
from app.rag.retriever import RetrievedChunk


def _chunk(text: str = "梯度下降的学习率过大会导致损失函数震荡。") -> RetrievedChunk:
    """随手造一块命中资料（分数拉满，工具层不做阈值过滤）。"""
    return RetrievedChunk(text=text, file_name="讲义.docx", page_no=1, score=0.9, group_id=1)


class _StubGateway:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.kwargs: dict = {}

    async def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.reply


class _NoCall:
    async def generate(self, **kwargs):
        raise AssertionError("该路径严禁调用 LLM")


# ===== 安全求值 =====


def test_safe_eval_arith_basic():
    """四则/幂/括号/全角数字：确定性正确（这是不让 7B 心算的全部理由）。"""
    assert tools.safe_eval_arith("23*47+108") == 1189
    assert tools.safe_eval_arith("(1+2)**3") == 27
    assert tools.safe_eval_arith("100/4") == 25
    # 全角经 extract 转半角后同样可算
    expr = tools.extract_arith("（２３）＊４７")
    assert expr is not None
    assert tools.safe_eval_arith(expr) == 1081


def test_safe_eval_rejects_malicious():
    """★恶意表达式结构级拒绝：名字/属性/调用/导入全部 ValueError（AST 白名单无绕过面）。"""
    for evil in (
        "__import__('os').system('dir')",
        "(1).__class__",
        "open('/etc/passwd')",
        "[x for x in range(10)]",
        "lambda: 1",
    ):
        with pytest.raises(ValueError):
            tools.safe_eval_arith(evil)


def test_safe_eval_rejects_overflow_and_divzero():
    """大幂/除零拒绝——「合法但灾难」的表达式不许进计算。"""
    with pytest.raises(ValueError):
        tools.safe_eval_arith("2**999999")
    with pytest.raises(ZeroDivisionError):
        tools.safe_eval_arith("1/0")


def test_extract_arith_boundaries():
    """整句纯算式才旁路：引导词可剥离，含语义的一律返回 None 走常规路由。"""
    assert tools.extract_arith("23*47+108等于多少") == "23*47+108"
    assert tools.extract_arith("计算 100/3") == "100/3"
    assert tools.extract_arith("第3页公式f(x)=3x+2在x=5时等于多少") is None  # 含字母语义
    assert tools.extract_arith("2026年考研时间") is None  # 无运算符
    assert tools.extract_arith("梯度下降是什么") is None


# ===== 试题解析 =====


_VALID_QUIZ = {
    "type": "quiz",
    "questions": [
        {"type": "choice", "question": "Q1", "options": ["A. x", "B. y"], "answer": "A", "explanation": "e"},
        {"type": "short", "question": "Q2", "answer": "标准", "explanation": "e2"},
    ],
}


def test_parse_quiz_valid_and_fenced():
    """合法 JSON 可解析；模型手痒包 ```json 围栏也能剥掉解析。"""
    plain = json.dumps(_VALID_QUIZ, ensure_ascii=False)
    assert tools.parse_quiz(plain) is not None
    fenced = f"```json\n{plain}\n```"
    assert tools.parse_quiz(fenced) is not None


@pytest.mark.parametrize(
    "bad",
    [
        "完全不是JSON",
        '{"type":"essay","questions":[]}',  # 类型不对
        '{"type":"quiz","questions":[]}',  # 空题列表
        '{"type":"quiz","questions":[{"type":"choice","question":"Q"}]}',  # 缺 answer/options
    ],
)
def test_parse_quiz_rejects_bad(bad):
    """半残/错构 JSON 一律 None（调用方走兜底，绝不渲染报错卡片）。"""
    assert tools.parse_quiz(bad) is None


# ===== qa / quiz / summary / calc =====


@pytest.mark.asyncio
async def test_qa_tool_normal_and_insufficient(monkeypatch):
    """正常：净化+闸门+拼来源；不足：统一兜底口径。"""
    monkeypatch.setattr(
        "app.agent.tools.gateway", _StubGateway("学习率过大会震荡。【来源：假.pdf，第9页】")
    )
    out = await tools.qa_tool("学习率", [_chunk()])
    assert out["hit"] is True and out["intent"] == "qa"
    assert "编造" not in out["answer"] and "【来源" in out["answer"]
    assert out["sources"][0]["file_name"] == "讲义.docx"

    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("根据现有资料无法回答。"))
    out = await tools.qa_tool("学习率", [_chunk()])
    assert out["hit"] is False
    assert out["answer"] == FALLBACK_MESSAGE
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_quiz_tool_ok_and_parse_fail(monkeypatch):
    """成功：answer 是可回读的规范 JSON + sources 独立承载溯源（不拼进 JSON）；
    解析失败：统一兜底（不返回半残 JSON）。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway(json.dumps(_VALID_QUIZ, ensure_ascii=False)))
    out = await tools.quiz_tool("出2道题", [_chunk()])
    assert out["hit"] is True and out["intent"] == "quiz"
    assert json.loads(out["answer"])["type"] == "quiz"  # answer 可直接回读
    assert "【来源" not in out["answer"]  # 绝不污染 JSON 结构
    assert out["sources"]  # 溯源由 sources 字段承载

    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("我来给你出题：第一题……"))
    out = await tools.quiz_tool("出2道题", [_chunk()])
    assert out["hit"] is False and out["answer"] == FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_summary_tool_insufficient_falls_back(monkeypatch):
    """总结同样受资料不足闸门约束（不知道就不给要点、不拼来源）。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("根据现有资料无法回答。"))
    out = await tools.summary_tool("总结这一节", [_chunk()])
    assert out["hit"] is False and out["answer"] == FALLBACK_MESSAGE
    assert out["intent"] == "summary"


@pytest.mark.asyncio
async def test_calc_tool_deterministic_value(monkeypatch):
    """纯算式：结果由 AST 定（1189），LLM 只生成步骤——stub 返回空也不影响结果正确。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("第一步：先乘后加。"))
    out = await tools.calc_tool("23*47+108等于多少")
    assert out["hit"] is True and out["intent"] == "calc"
    assert "1189" in out["answer"]
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_calc_tool_non_arith_with_chunks_delegates_qa(monkeypatch):
    """分类为 calc 但整句非纯算式（如「算一下第3页例题」）→ 有资料退回答疑，路由不落空。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("根据例题，答案是……【来源】"))
    out = await tools.calc_tool("帮我算一下那道例题", [_chunk()])
    assert out["intent"] == "qa"  # 已退化为答疑


# ===== 判分 =====


@pytest.mark.asyncio
async def test_score_all_choice_zero_llm(monkeypatch):
    """全单选卷：代码层精确判分，零 LLM 调用（能确定的绝不交给模型）。"""
    monkeypatch.setattr("app.agent.tools.gateway", _NoCall())
    questions = [
        {"type": "choice", "question": "Q1", "options": ["A", "B"], "answer": "A"},
        {"type": "choice", "question": "Q2", "options": ["A", "B"], "answer": "B"},
    ]
    out = await tools.score_quiz(questions, ["A", "A"])  # 对1错1
    assert out["score"] == 50


@pytest.mark.asyncio
async def test_score_mixed_weights_llm_short_answer(monkeypatch):
    """混合卷：单选代码判 + 简答 LLM 判，总分等权合成。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway('{"score":80,"comment":"要点基本齐全"}'))
    questions = [
        {"type": "choice", "question": "Q1", "options": ["A", "B"], "answer": "A"},
        {"type": "short", "question": "Q2", "answer": "要点ABC"},
    ]
    out = await tools.score_quiz(questions, ["A", "我的作答"])
    assert out["score"] == 90  # (100 + 80) / 2
    assert out["comment"] == "要点基本齐全"


@pytest.mark.asyncio
async def test_score_llm_garbage_raises(monkeypatch):
    """LLM 判分输出解析失败 → RuntimeError（接口层翻译 502），绝不返回假分数。"""
    monkeypatch.setattr("app.agent.tools.gateway", _StubGateway("这卷子没法判……"))
    questions = [{"type": "short", "question": "Q", "answer": "标准"}]
    with pytest.raises(RuntimeError):
        await tools.score_quiz(questions, ["作答"])
