# -*- coding: utf-8 -*-
"""
意图分类（M4）：把用户提问归到四个单标签之一，供 LangGraph 路由。

为什么用「LLM 单标签分类」而不是关键词规则：
1. 「出3道题」「聊聊上一节讲了啥」这类意图表达千变万化，关键词白名单永远漏
   （风险预案 R3 的立项理由）；
2. 单标签选择是 7B 最稳的任务形态——温度 0、强制枚举、输出≤16 token，
   实测稳定性远高于开放式生成；
3. 仍然保留「非法输出默认 qa」的兜底：模型再稳也可能抽风多吐标点，
   路由层绝不允许出现「没有意图」的卡死状态。

为什么分类放在「检索命中之后」（见 graph 布局）：
空召回要走「零 LLM 调用」的防幻觉硬闸门——分类本身也是一次 LLM 调用，
若排在检索前，那条答辩核心卖点的字面属性（测试断言 calls==0）就被破坏了。
计算意图是例外：纯算式在 graph 入口被正则旁路先接走，不依赖检索。
"""
from app.core.config import settings
from app.core.llm import gateway  # ★测试补丁缝：monkeypatch app.agent.intent.gateway

# 合法意图标签（唯一权威定义，graph 路由与测试都从这里 import）
INTENT_QA = "qa"
INTENT_QUIZ = "quiz"
INTENT_SUMMARY = "summary"
INTENT_CALC = "calc"
VALID_INTENTS = (INTENT_QA, INTENT_QUIZ, INTENT_SUMMARY, INTENT_CALC)

# 分类调用输出上限：单标签最多几个字符，16 token 绰绰有余（红线：输出受控）
_CLASSIFY_NUM_PREDICT = 16

# 分类 prompt：枚举 + few-shot + 明确「拿不准输出 qa」
_SYSTEM = (
    "你是意图分类器，只输出一个标签，禁止输出任何其他文字。\n"
    "可选标签：qa（知识答疑）、quiz（出题/考试/随堂测）、"
    "summary（总结/概括/回顾）、calc（纯数值计算）。\n"
    "示例：「洛必达法则什么时候用」→qa；「出5道关于导数的题」→quiz；"
    "「总结这一章的要点」→summary；「125*8等于多少」→calc。\n"
    "拿不准时输出 qa。"
)


def parse_intent(raw: str) -> str:
    """把模型输出解析成合法标签；任何不认识的输出一律回退 qa（路由永不卡死）。"""
    text = (raw or "").strip().lower()
    # 精确命中优先（最常见：模型乖巧地只吐了标签）
    if text in VALID_INTENTS:
        return text
    # 宽松命中：模型可能输出「quiz（出题）」这类带尾巴的形式——按标签子串找，
    # 顺序按长度倒序防止 "qa" 在其他标签文本里被误抢（如 "quiz" 含 "ui" 但不含 "qa" 子串规则，
    # 这里仍按完整单词标签做子串匹配，四个标签互不为子串，顺序无实质影响，倒序仅为稳妥）
    for label in sorted(VALID_INTENTS, key=len, reverse=True):
        if label in text:
            return label
    return INTENT_QA  # 非法/空输出 → 默认答疑


async def classify_intent(question: str) -> str:
    """LLM 单标签意图分类，返回 VALID_INTENTS 之一。

    temperature=0 + num_predict=16：确定性任务的标准姿势（红线：输出受控）。
    """
    raw = await gateway.generate(
        model=settings.llm_model,
        prompt=question,
        system=_SYSTEM,
        num_predict=_CLASSIFY_NUM_PREDICT,
        temperature=0.0,
    )
    return parse_intent(raw)
