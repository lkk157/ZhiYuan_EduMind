# -*- coding: utf-8 -*-
"""
向量库抽象与 ChromaDB 实现（读/写路径共用）。

为什么要有 VectorStore 接口这一层（答辩可直接讲）：
MySQL 是事实源，向量库只是「衍生索引」——衍生组件必须可整体替换。
生产方案（README 对比表）是 Qdrant 独立向量服务：届时只新增一个 QdrantStore
实现类，retriever / ingest / api 上层调用零改动，本文件的 ChromaStore 退居本地降级方案。

两条防翻车红线（本文件的命门）：
1. ★ 绝不允许 chroma 默认 embedding function 被触发——它会在首次调用时
   联网下载 onnx 小模型，答辩机断网即翻车。所以 collection 创建时显式
   `embedding_function=None`，且 add/query 一律显式传 embeddings / query_embeddings；
2. chroma 对 n_results 超过集合条数、以及 n_results=0 的行为不稳定
   （实测 1.5.9 对 0 直接抛错），所以查询前先 count、再把 n_results 夹到 [1, count]。
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from chromadb import PersistentClient

from app.core.config import PROJECT_ROOT, settings
from app.core.exceptions import UpstreamError
from app.rag import embeddings


@dataclass
class SearchHit:
    """一条检索命中：score = 1 - cosine 距离（即余弦相似度，越大越相关）。

    为什么 score 统一成「越大越像」：上层阈值过滤（score >= threshold）只需
    一个方向的比较，不必在各处反复换算距离/相似度。
    """

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """向量库最小接口。

    生产换 Qdrant 只换实现类（新增 QdrantStore 实现本接口即可），
    上层（retriever / ingest）面向本接口编程，零改动——这是「接口与实现分离」的落点。
    """

    def query(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """按向量检索 top_k 条命中（score=1-余弦距离）。"""
        ...

    def get_all(self) -> list[SearchHit]:
        """拉取全量块（id/text/metadata，不含向量）——混合检索的 BM25 语料来源。"""
        ...

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """按 id 取回存量向量（BM25 独家命中补算余弦分数用）。"""
        ...

    def delete(self, ids: list[str]) -> Any:
        """按向量 id 删除（重传增量入库时清掉被移除/被改动的旧块）。"""
        ...

    def drop(self) -> Any:
        """删除整个 collection（删除知识分组时级联清空向量库）。"""
        ...

    async def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """写入/覆盖若干块（id 稳定则覆盖同块，支撑「改一块只重算一块」）。

        为什么是 async：契约参数表只有 ids/texts/metadatas（没有 embeddings），
        向量必须在内部经 gateway 现算并显式传给 chroma（红线 1），
        而 gateway 是异步 I/O，故本方法只能是协程——与 async ingest_file 同属写路径异步世界。
        """
        ...


# 进程级共享的持久化客户端缓存（懒加载）。
# 为什么缓存：PersistentClient 会打开磁盘目录，每次 store_for 都新建一个
# 既浪费文件句柄，又会触发 chroma 的多客户端告警；单测显式传 EphemeralClient，
# 完全不走这条路径。
_default_client_cache: Any = None


def _get_default_client() -> Any:
    """懒加载持久化 Chroma 客户端（生产/演示用，数据落 settings.chroma_dir）。

    为什么相对路径按项目根解析：settings.chroma_dir 按约定是「相对项目根目录」
    （见 app/core/config.py），若按 cwd 解析，从不同目录启动服务会把向量库写得到处都是。
    """
    global _default_client_cache
    if _default_client_cache is None:
        chroma_dir = settings.chroma_dir
        if not chroma_dir.is_absolute():
            chroma_dir = PROJECT_ROOT / chroma_dir
        _default_client_cache = PersistentClient(path=str(chroma_dir))
    return _default_client_cache


class ChromaStore:
    """ChromaDB 实现：按 collection 隔离（一个用户-分组一个 collection）。

    为什么一个分组一个 collection：分组隔离是权限边界（越权检索=数据泄漏），
    物理隔离比 where 过滤更不容易写漏。
    """

    def __init__(self, collection: str, client: Any = None):
        """绑定（必要时创建）collection。

        client 缺省用持久化客户端；单测传 chromadb.EphemeralClient，磁盘零接触。
        """
        self._name = collection
        self._client = client if client is not None else _get_default_client()
        # ★ 红线 1：embedding_function=None 显式禁用默认 onnx EF（断网环境致命）；
        # cosine 空间与 score = 1 - distance 的换算约定一致。
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    async def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """向量化并写入/覆盖若干块（同 id 覆盖 = 增量更新单块）。

        为什么向量在这里现算：契约的 upsert 参数表不含 embeddings，
        而红线 1 要求显式传 embeddings 给 chroma——所以向量化只能在方法内部完成；
        只对传入的块做向量化，天然满足「改一块只重算一块」的增量承诺。
        """
        if not ids:
            return  # 空批次直接短路，省一次 Ollama 往返
        # 走 app.rag.embeddings 模块属性调用：单测 monkeypatch 一个点即全局生效
        vectors = await embeddings.embed_texts(texts)
        if len(vectors) != len(ids):
            # Ollama 返回条数不一致属于上游异常，必须炸出来而不是静默入库错位
            raise UpstreamError("向量化结果数量与文本数量不一致，请重试")
        # ★ 红线 1：显式 embeddings= / documents=，绝不让 chroma 自己去 embed
        self._col.upsert(
            ids=list(ids),
            embeddings=vectors,
            documents=list(texts),
            metadatas=list(metadatas),
        )

    def query(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """按向量检索 top_k 条命中，score = 1 - 余弦距离。

        为什么 n_results 要夹逼：chroma 对 n_results 超过集合条数行为不稳，
        对 n_results=0 直接抛错（实测 1.5.9）；空集合则直接返回 [] 不下发查询。
        """
        count = self._col.count()
        if count == 0 or top_k <= 0:
            return []
        n_results = min(top_k, count)  # 绝不超过集合内条数
        # ★ 红线 1：只传 query_embeddings=，绝不传 query_texts=（会触发默认 EF）
        res = self._col.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        id_rows = res.get("ids") or [[]]
        doc_rows = res.get("documents") or [[]]
        meta_rows = res.get("metadatas") or [[]]
        dist_rows = res.get("distances") or [[]]
        hits: list[SearchHit] = []
        for id_, text, meta, dist in zip(
            id_rows[0], doc_rows[0], meta_rows[0], dist_rows[0]
        ):
            hits.append(
                SearchHit(
                    id=id_,
                    text=text or "",
                    # cosine 空间下：相似度 = 1 - 距离（距离 0 = 完全相同）
                    score=1.0 - float(dist),
                    metadata=dict(meta or {}),
                )
            )
        return hits

    def get_all(self) -> list[SearchHit]:
        """拉取全量块（含 text/metadata，不含向量）——BM25 语料一次取齐。

        为什么可以全量拉：语料规模 = 文档块数（毕设体量几千条，毫秒级）；
        score 置 0.0 占位——BM25 路的分数由 BM25 自己算，这里只提供文本。
        为什么不动 MySQL：块正文只存在于向量库（MySQL 只存指纹），
        从这里取零表结构变更（质量包的「不加表」承诺）。
        """
        count = self._col.count()
        if count == 0:
            return []
        res = self._col.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            SearchHit(id=i, text=d or "", score=0.0, metadata=dict(m or {}))
            for i, d, m in zip(ids, docs, metas)
        ]

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """按 id 取回存量向量 → {id: vector}（缺失的 id 不出现在结果里）。

        用途唯一：BM25 独家捞回的块没有查询时的余弦分，取其存量向量
        与查询向量现算余弦——保证 sources 里的「相似度」字段对两类命中同口径。
        """
        if not ids:
            return {}
        res = self._col.get(ids=list(ids), include=["embeddings"])
        out: dict[str, list[float]] = {}
        # ★ 不能写 `res.get("embeddings") or []`（2026-09-25 生产 500 实测踩坑）：
        # chroma 对 embeddings 返回 numpy 二维数组，ndarray 的真值判断对多元素
        # 数组直接抛「truth value is ambiguous」——ids/documents 是 Python list
        # 才能用 `or []`，embeddings 必须显式判 None 后逐元素转 list。
        embeddings = res.get("embeddings")
        if embeddings is None:
            return out
        for id_, emb in zip(res.get("ids") or [], list(embeddings)):
            if emb is not None:
                out[id_] = [float(x) for x in emb]
        return out

    def delete(self, ids: list[str]) -> None:
        """按向量 id 删除旧块（增量入库时清掉被移除/被改动的块）。"""
        if ids:
            self._col.delete(ids=list(ids))

    def drop(self) -> None:
        """删除整个 collection（删除知识分组时级联清空，不留孤儿索引）。"""
        self._client.delete_collection(self._name)


def store_for(user_id: int, group_id: int, client: Any = None) -> ChromaStore:
    """取「用户-分组」对应的向量库（collection 名 u{user_id}g{group_id}）。

    为什么命名里带 user_id：collection 名是权限边界的一部分，
    肉眼可审计（答辩演示时一眼看出这是谁的哪个分组），也杜绝跨用户误用同名集合。
    """
    return ChromaStore(collection=f"u{user_id}g{group_id}", client=client)
