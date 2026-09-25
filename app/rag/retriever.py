# -*- coding: utf-8 -*-
"""
检索器（读路径核心）：问题向量化 → 跨分组检索 → 合并排序 → 阈值过滤 → top_k 截断。

为什么检索逻辑独立成层（api 不许自己拼查询）：
1. 分组隔离是权限边界：只查调用者传入的 group_ids 对应的 collection，
   api 层漏写过滤就等于数据泄漏，收口在这里只写一次；
2. 「阈值过滤 + 截断」是防幻觉的第一道闸——召回质量不达标宁可空结果，
   由调用方走 FALLBACK_MESSAGE 兜底，也好过把无关资料喂给 LLM 编造答案；
3. 单测通过 stores 参数注入假向量库，不碰磁盘、不碰 Chroma 单例。
"""
from dataclasses import dataclass

from app.core.config import settings
from app.rag import embeddings
from app.rag.vector_store import ChromaStore, store_for


@dataclass
class RetrievedChunk:
    """一块召回资料（带溯源信息）。

    file_name + page_no 是溯源根基（最终答案末尾的【来源：文件名，第X页】就来自这里），
    score 用于阈值过滤与排序，group_id 标明资料归属分组。
    """

    text: str
    file_name: str
    page_no: int
    score: float
    group_id: int


async def retrieve(
    question: str,
    *,
    user_id: int,
    group_ids: list[int],
    stores: dict | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    page_range: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> list[RetrievedChunk]:
    """按问题检索若干分组下的相关资料块，按 score 降序返回。

    参数：
    - group_ids: 本次允许检索的知识分组（权限边界，只查这些分组的 collection）；
    - stores: {group_id: ChromaStore}，供单测注入假向量库；缺省用 store_for 现建；
    - top_k / threshold: 缺省取 settings.top_k / settings.score_threshold（阈值在 M3 标定后回填 .env）；
    - page_range / file_name: 范围过滤（M4 质量优化第 2 层，来自 agent/scope.py 的
      「第X-Y页」「第N章」解析）——命中块按页码区间/文件名先过滤再过阈值，
      用户指定了范围就绝不把范围外的块喂给模型（总结越界的根治）。

    返回空列表表示「没召回到达标资料」——本层不编兜底话术，
    由调用方（/chat/ask）决定走 FALLBACK_MESSAGE 且不调 LLM（防幻觉兜底）。
    """
    # 取代配置：未显式指定就用全局阈值/条数（配置只从 settings 读，禁止硬编码）
    k = settings.top_k if top_k is None else top_k
    min_score = settings.score_threshold if threshold is None else threshold

    # 空问题/空分组/非法 top_k 直接短路：省一次 Ollama 向量化往返（显存红线：不做无谓模型调用）
    if not question.strip() or not group_ids or k <= 0:
        return []

    # 问题向量化：走 app.rag.embeddings 的模块属性调用，保证单测 monkeypatch 一个点生效
    vectors = await embeddings.embed_texts([question])
    if not vectors:
        return []
    vector = vectors[0]

    # stores 缺省按 group_ids 现建；显式传入时只用注入的实例（单测零磁盘）
    if stores is None:
        store_map: dict[int, ChromaStore] = {gid: store_for(user_id, gid) for gid in group_ids}
    else:
        store_map = stores

    collected: list[RetrievedChunk] = []
    for gid in group_ids:
        store = store_map.get(gid)
        if store is None:
            continue  # 注入的 stores 缺某分组就跳过（不悄悄建真库）
        for hit in store.query(vector, k):
            meta = hit.metadata or {}
            collected.append(
                RetrievedChunk(
                    text=hit.text,
                    # 溯源三元组一律从 metadata 取（入库时写入的键名是层间契约）
                    file_name=str(meta.get("file_name", "")),
                    page_no=int(meta.get("page_no", 0) or 0),
                    score=hit.score,
                    group_id=int(meta.get("group_id", gid) or gid),
                )
            )

    # 范围过滤放在排序/阈值之前：先按用户指定的页码区间/文件收窄候选，
    # 再在范围内做阈值+截断——顺序反了会把范围外的高分块截进来
    if page_range is not None:
        lo, hi = page_range
        collected = [c for c in collected if lo <= c.page_no <= hi]
    if file_name:
        collected = [c for c in collected if c.file_name == file_name]

    # 合并各分组命中后统一排序：score 降序（同分稳定，命中顺序可复现，便于测试断言）
    collected.sort(key=lambda c: c.score, reverse=True)
    # 先过阈值（宁缺毋滥，防幻觉）再截断 top_k——顺序不能反，否则会把低分块挤掉高分块
    passed = [c for c in collected if c.score >= min_score]
    return passed[:k]
