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

    # ===== RAG 质量包：混合检索分支（RETRIEVAL_MODE=hybrid 时启用）=====
    # 放在向量路径之前分发：两条路完全独立，vector 模式一个字节都不多跑
    if settings.retrieval_mode == "hybrid":
        return _retrieve_hybrid(
            question,
            vector,
            store_map,
            group_ids=group_ids,
            k=k,
            threshold=min_score,
            page_range=page_range,
            file_name=file_name,
        )

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


def _in_scope(meta: dict, page_range: tuple[int, int] | None, file_name: str | None) -> bool:
    """范围过滤谓词（两条检索路径共用）：页码区间 + 文件名。"""
    if page_range is not None:
        lo, hi = page_range
        try:
            page = int(meta.get("page_no", 0) or 0)
        except (TypeError, ValueError):
            return False
        if not lo <= page <= hi:
            return False
    if file_name and str(meta.get("file_name", "")) != file_name:
        return False
    return True


def _retrieve_hybrid(
    question: str,
    vector: list[float],
    store_map: dict,
    *,
    group_ids: list[int],
    k: int,
    threshold: float,
    page_range: tuple[int, int] | None,
    file_name: str | None,
) -> list[RetrievedChunk]:
    """BM25 + 向量双路召回 → 范围过滤 → 双闸准入 → RRF 融合（纯同步 CPU 计算）。

    为什么语料/向量候选都收进同一个 texts/metas 字典：融合后 final_ids 里
    可能出现「只有 BM25 见过」的块，组装 RetrievedChunk 时必须能查到它的
    文本与 metadata——统一以 id 为键的池子保证两路候选可互查。

    分数口径：向量路自带余弦分；BM25 独家命中用存量向量现算余弦
    （get_embeddings + hybrid.cosine）——sources 里的「相似度」对两类命中同口径。
    """
    from app.rag import hybrid  # 函数内导入：纯 vector 模式不触碰混合检索模块

    limit = k * hybrid._CANDIDATE_MULTIPLIER

    # ★ 权限边界：只许遍历 group_ids 与 store_map 的交集——group_ids 才是授权权威，
    # store_map 可能比它宽（单测注入 superset、或未来调用方传宽）。
    # 向量路原本按 group_ids 循环天然安全，hybrid 第一版按 store_map 全键循环，
    # 被 test_group_isolation 当场抓住（跨组捞语料=越权检索），已改为交集遍历。
    active_stores = {gid: store_map[gid] for gid in group_ids if gid in store_map}

    texts: dict[str, str] = {}
    metas: dict[str, dict] = {}
    group_of: dict[str, int] = {}
    vector_scores: dict[str, float] = {}
    vector_ranked: list[str] = []

    # 1) 向量路候选（每组取 top k*2，给融合留放量）
    for gid, store in active_stores.items():
        for hit in store.query(vector, limit):
            if hit.id in vector_scores:
                continue  # id 全局唯一（doc_id:chunk_index），跨组重入只记一次
            vector_scores[hit.id] = hit.score
            vector_ranked.append(hit.id)
            texts[hit.id] = hit.text
            metas[hit.id] = hit.metadata
            group_of[hit.id] = gid

    # 2) BM25 语料 = 全池（向量候选本就在 texts 池里；再拉全量补上向量没召回的块——
    #    「关键词强命中但余弦擦边」的块只可能从 get_all 这条路进候选）
    for gid, store in active_stores.items():
        for item in store.get_all():
            if item.id in texts:
                continue
            texts[item.id] = item.text
            metas[item.id] = item.metadata
            group_of[item.id] = gid
    corpus_ids = list(texts.keys())  # 插入序稳定，融合结果可复现
    corpus_texts = [texts[i] for i in corpus_ids]

    bm25_ranked: list[str] = []
    bm25_scores: dict[str, float] = {}
    index = hybrid.build_bm25(corpus_texts)
    if index is not None and corpus_ids:
        query_scores = index.get_scores(hybrid.tokenize(question))
        ranked = sorted(zip(corpus_ids, query_scores), key=lambda x: -x[1])
        for id_, score in ranked[:limit]:
            if float(score) > 0:  # 0 分 = 无任何词交集，不可能过准入下限
                bm25_ranked.append(id_)
                bm25_scores[id_] = float(score)

    # 3) 范围过滤作用于两路名次表（先收窄候选，再谈准入与融合）
    vector_ranked = [i for i in vector_ranked if _in_scope(metas.get(i, {}), page_range, file_name)]
    bm25_ranked = [i for i in bm25_ranked if _in_scope(metas.get(i, {}), page_range, file_name)]

    # 4) 双闸准入 + RRF 融合，截 top_k
    final_ids = hybrid.fuse(
        vector_ranked,
        bm25_ranked,
        vector_scores,
        bm25_scores,
        threshold=threshold,
        floor=settings.bm25_floor,
    )[:k]

    # 5) BM25 独家命中补余弦分（同口径展示相似度）
    missing = [i for i in final_ids if i not in vector_scores]
    if missing:
        by_group: dict[int, list[str]] = {}
        for id_ in missing:
            by_group.setdefault(group_of[id_], []).append(id_)
        for gid, ids in by_group.items():
            store = store_map.get(gid)
            if store is None:
                continue
            for id_, emb in store.get_embeddings(ids).items():
                vector_scores[id_] = hybrid.cosine(vector, emb)
        # 取不到存量向量的（数据异常）按 0.0 分兜底展示——宁可显示 0.00 也不丢命中

    # 6) 组装结果（与向量路同构的 RetrievedChunk，上层零感知差异）
    result: list[RetrievedChunk] = []
    for id_ in final_ids:
        meta = metas.get(id_, {})
        result.append(
            RetrievedChunk(
                text=texts.get(id_, ""),
                file_name=str(meta.get("file_name", "")),
                page_no=int(meta.get("page_no", 0) or 0),
                score=vector_scores.get(id_, 0.0),
                group_id=int(meta.get("group_id", group_of.get(id_, 0)) or group_of.get(id_, 0)),
            )
        )
    return result
