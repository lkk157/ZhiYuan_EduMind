# -*- coding: utf-8 -*-
"""
LangGraph 状态图（M4）：Agent 路由中枢——改写 → 检索 → 分类 → 工具分发。

图布局（为什么是这个顺序，答辩可直接照讲）：

    START → start(计算旁路/改写) ──算式──> calc ──> END
                 │非算式
                 v
            retrieve(检索) ──空召回──> fallback ★零 LLM 调用（防幻觉硬闸门）
                 │命中
                 v
            classify(意图分类) ──> qa | quiz | summary | calc ──> END

三条设计红线：
1. **检索前置**：分类/改写都可能产生 LLM 调用，空召回必须在它们全跑完之前
   （改写除外——无历史时不调用，而空召回场景的历史仅在「有会话」时存在，
   单测的零调用断言以无历史路径为准，语义见 short_term.rewrite_question）；
   分类放检索后，「无关问题一个模型都不碰」的字面属性保住了；
2. **计算旁路在最前**：纯算式往往检索不到课件（会误入空召回兜底），
   正则先接走，计算工具不依赖知识库；
3. **非法意图默认 qa**（intent.parse_intent 内兜底），路由表里四个标签都有出口，
   状态图永远不会走进死胡同。

为什么用 LangGraph 而不是 if/elif：技术栈锁定（CLAUDE.md §4）；
状态图节点/条件边在论文里就是架构图本身；后续加工具（M5 记忆召回）= 加节点，
不动既有路由。
"""
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent import intent, tools
from app.core.config import settings
from app.memory import short_term
from app.rag.retriever import retrieve  # ★测试补丁缝：monkeypatch app.agent.graph.retrieve

# 图编译一次、进程内复用：每次请求重新 compile 会白白重建节点图（无状态图，复用安全）
_graph = None


class AgentState(TypedDict, total=False):
    """图状态：各节点只写自己负责的键，LangGraph 按键合并（partial return 合法）。"""

    # 输入侧
    question: str
    user_id: int
    group_ids: list[int]
    history: list[dict]
    guide_mode: bool
    # 范围过滤（agent/scope.py 在接口层解析后传入，图内只负责透传给检索）
    page_range: tuple[int, int] | None
    file_name: str | None
    scope_note: str | None
    # 节点产物
    intent: str
    rewritten: str
    chunks: list  # list[RetrievedChunk]（内存图，任意对象直接流转）
    # 输出侧
    answer: str
    sources: list[dict]
    hit: bool


async def _node_start(state: AgentState) -> dict:
    """入口：先试计算旁路（纯正则零 LLM），否则做 query 改写。"""
    question = state["question"]
    # 旁路命中 → 打上 calc 意图标记，条件边直送 calc 节点（不检索、不分类）
    if tools.extract_arith(question) is not None:
        return {"intent": intent.INTENT_CALC}
    # 无历史时 rewrite_question 是纯函数（零模型调用）——防幻觉零调用路径依赖此行为
    rewritten = await short_term.rewrite_question(question, state.get("history") or [])
    return {"rewritten": rewritten, "intent": intent.INTENT_QA}  # intent 先给默认值


async def _node_retrieve(state: AgentState) -> dict:
    """检索（改写后的问题 + 范围过滤）：阈值/截断在 retriever 内，空列表=没达标资料。

    检索池按三工具 top_k 的最大值召回：各工具随后各自截断（出题聚焦/总结更全），
    池子不够大时「总结」会先被默认 top_k 削掉材料——取 max 保证谁都不缺料。
    """
    query = state.get("rewritten") or state["question"]
    pool_k = max(settings.top_k, settings.top_k_quiz, settings.top_k_summary)
    chunks = await retrieve(
        query,
        user_id=state["user_id"],
        group_ids=state["group_ids"],
        top_k=pool_k,
        page_range=state.get("page_range"),
        file_name=state.get("file_name"),
    )
    return {"chunks": chunks}


async def _node_classify(state: AgentState) -> dict:
    """意图分类（仅检索命中后才走到这里——见模块 docstring 红线 1）。"""
    label = await intent.classify_intent(state["question"])
    return {"intent": label}


async def _node_fallback(state: AgentState) -> dict:
    """空召回兜底：固定话术、空来源、hit=false——本节点严禁调用任何 LLM。"""
    return tools._fallback(state.get("intent") or intent.INTENT_QA)


async def _node_qa(state: AgentState) -> dict:
    return await tools.qa_tool(
        state["question"], state["chunks"], guide_mode=bool(state.get("guide_mode"))
    )


async def _node_quiz(state: AgentState) -> dict:
    return await tools.quiz_tool(state["question"], state["chunks"])


async def _node_summary(state: AgentState) -> dict:
    return await tools.summary_tool(state["question"], state["chunks"])


async def _node_calc(state: AgentState) -> dict:
    # chunks 可能不存在（旁路路径没检索）——分类路径才有
    return await tools.calc_tool(state["question"], state.get("chunks"))


def _route_after_start(state: AgentState) -> str:
    """入口条件边：算式 → calc；否则 → 检索。"""
    if state.get("intent") == intent.INTENT_CALC:
        return "calc"
    return "retrieve"


def _route_after_retrieve(state: AgentState) -> str:
    """检索条件边：空召回 → 兜底（零 LLM）；命中 → 意图分类。"""
    return "classify" if state.get("chunks") else "fallback"


def _route_after_classify(state: AgentState) -> str:
    """分类条件边：四标签各有节点；parse_intent 已保证标签合法（非法→qa）。"""
    label = state.get("intent", intent.INTENT_QA)
    return label if label in intent.VALID_INTENTS else intent.INTENT_QA


def get_graph():
    """惰性编译状态图（进程内单例）。"""
    global _graph
    if _graph is None:
        builder = StateGraph(AgentState)
        builder.add_node("start", _node_start)
        builder.add_node("retrieve", _node_retrieve)
        builder.add_node("classify", _node_classify)
        builder.add_node("fallback", _node_fallback)
        builder.add_node("qa", _node_qa)
        builder.add_node("quiz", _node_quiz)
        builder.add_node("summary", _node_summary)
        builder.add_node("calc", _node_calc)

        builder.add_edge(START, "start")
        builder.add_conditional_edges(
            "start", _route_after_start, {"calc": "calc", "retrieve": "retrieve"}
        )
        builder.add_conditional_edges(
            "retrieve",
            _route_after_retrieve,
            {"classify": "classify", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "classify",
            _route_after_classify,
            {label: label for label in intent.VALID_INTENTS},
        )
        # 四个工具节点 + 兜底节点全部直连 END（线性收尾，无二次路由）
        for node in ("calc", "fallback", "qa", "quiz", "summary"):
            builder.add_edge(node, END)
        _graph = builder.compile()
    return _graph


async def run_agent(
    *,
    question: str,
    user_id: int,
    group_ids: list[int],
    history: list[dict],
    guide_mode: bool = False,
    page_range: tuple[int, int] | None = None,
    file_name: str | None = None,
    scope_note: str | None = None,
) -> dict[str, Any]:
    """跑一轮 Agent，返回 {answer, sources, hit, intent, scope_note} 契约。

    薄封装的意义：接口层只认这一个函数，图的编译/状态构造细节全部内聚在本模块；
    返回前做键级兜底——图内任一节点漏写字段时，宁可给兜底值也不让 KeyError 变 500。
    scope_note：范围解析失败的人话提示（如「课件无目录请用页码提问」），
    命中/兜底都原样带给前端展示；解析成功或无范围词时为 None。
    """
    final: AgentState = await get_graph().ainvoke(
        {
            "question": question,
            "user_id": user_id,
            "group_ids": group_ids,
            "history": history,
            "guide_mode": guide_mode,
            "page_range": page_range,
            "file_name": file_name,
            "scope_note": scope_note,
        }
    )
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources") or [],
        "hit": bool(final.get("hit")),
        "intent": final.get("intent") or intent.INTENT_QA,
        "scope_note": final.get("scope_note"),
    }
