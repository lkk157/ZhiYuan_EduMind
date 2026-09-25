# -*- coding: utf-8 -*-
"""
短期记忆（M4）：滑动窗口读取会话历史 + query 改写（指代消解）。

为什么要做 query 改写，而不是把历史原文直接塞进生成 prompt：
1. 检索是「指代消解」的主战场——「它呢？」「第二种呢？」直接拿去向量化，
   召回的全是字面相似而非语义所指；改写成独立完整问题后，检索一次就准；
2. 生成侧零膨胀：num_ctx≤4096 红线（CLAUDE.md §3）下，资料块已占大头，
   再叠历史原文必然挤爆上下文——改写调用是独立小请求（输出≤96 token），
   与生成互不干扰，窗口只在改写请求内部出现；
3. 数据源直接复用 M4 前已建好的 conversations/messages 两表——零新增表、零迁移。

为什么窗口按「条数」而不是「轮数」配置：消息表一问一答各一条，
条数是存储层自然粒度；6 条 = 最近 3 轮，语义在 config 注释里说明即可。
"""
import logging

from app.core.config import settings
from app.core.llm import gateway  # ★测试补丁缝：单测 monkeypatch app.memory.short_term.gateway
from app.db import crud

logger = logging.getLogger(__name__)

# 改写调用的输出上限：独立小请求，红线要求输出受控——一句独立问句远用不到 96 token
_REWRITE_NUM_PREDICT = 96

# 历史单条截断长度：防止用户发过超长文本把改写请求本身撑大
_HISTORY_ITEM_CHARS = 300


def load_window(db, *, user_id: int, conversation_id: int | None) -> list[dict]:
    """读会话最近 N 条**有效**消息，返回 [{"role","content"}, ...]（时间正序）。

    为什么先剔除兜底轮再截窗（2026-09-25 实测缺陷，M4 滑窗遗留、M6 demo 抓获）：
    hit=False 的一轮 = 固定拒答话术 + 一个超纲问题（如「番茄炒蛋要放多少盐」）——
    对改写零语义信息，却会把 query 改写带偏到无关话题，检索随之空召回，
    且污染随历史持续多轮（序列 正例→负例→出题 中出题连续翻车，实测复现两组对照）。
    因此只让「有实质内容的对话」参与改写；问答必须**成对**剔除——
    只删回答留提问没有用，问题文本照样带偏。

    先过滤后截窗：N 的语义从「最近 N 条消息」变为「最近 N 条有效消息」，
    中间夹几轮兜底也不会把更早的实质对话挤出窗口（指代消解仍找得到所指）。

    conversation_id=None（无状态问答）或剔除后为空 → 返回 []，
    调用方以「空历史」语义处理（不触发改写，省一次模型调用）。
    """
    if conversation_id is None:
        return []
    messages = crud.list_messages(db, user_id=user_id, conversation_id=conversation_id)

    # 成对剔除兜底轮：user 暂存等配对；assistant hit=False 时连同其提问一并作废
    kept = []
    pending_user = None
    for m in messages:
        if m.role == "user":
            pending_user = m
            continue
        if m.hit is False:  # 兜底回答：该轮整对作废（提问连坐，理由见 docstring）
            pending_user = None
            continue
        if pending_user is not None:
            kept.append(pending_user)
            pending_user = None
        kept.append(m)
    if pending_user is not None:
        kept.append(pending_user)  # 防御：孤儿提问（成对原子落库下不应出现）

    window = kept[-settings.short_term_window :] if settings.short_term_window > 0 else []
    return [
        {"role": m.role, "content": (m.content or "")[:_HISTORY_ITEM_CHARS]}
        for m in window
    ]


def build_rewrite_prompt(question: str, history: list[dict]) -> tuple[str, str]:
    """把「历史 + 当前问」编排成 (system, prompt)，让模型输出独立完整的检索问题。"""
    system = (
        "你是查询改写器。结合对话历史，把用户当前的提问改写成一个"
        "不依赖上下文、可独立理解的完整问句。只输出改写后的问句，"
        "不要解释、不要回答问题、不要添加任何前后缀。"
    )
    lines = []
    for item in history:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('content', '')}")
    history_text = "\n".join(lines) if lines else "（无历史）"
    prompt = f"对话历史：\n{history_text}\n\n当前提问：{question}\n改写后的问句："
    return system, prompt


async def rewrite_question(question: str, history: list[dict]) -> str:
    """有历史才改写，无历史原样返回（零模型调用）。

    改写失败/输出为空一律回退原问题——改写是「锦上添花」的增强，
    绝不能因为它挂掉就让提问整体失败（宁可用原问题检索，也好过 502）。
    """
    if not history:
        return question
    system, prompt = build_rewrite_prompt(question, history)
    try:
        raw = await gateway.generate(
            model=settings.llm_model,
            prompt=prompt,
            system=system,
            num_predict=_REWRITE_NUM_PREDICT,
            temperature=0.0,  # 改写要稳定可复现，温度 0
        )
    except Exception:
        # 网关异常（Ollama 挂了）也不阻断：后续检索/生成同样会失败并统一报错，
        # 这里吞掉只影响改写质量，不让错误栈从改写节点冒出来误导排查方向
        logger.exception("query 改写失败，使用原始问题")
        return question
    rewritten = (raw or "").strip()
    return rewritten if rewritten else question
