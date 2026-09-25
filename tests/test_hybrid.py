# -*- coding: utf-8 -*-
"""
RAG 质量包单测：RRF 融合序、双闸准入语义、BM25 精确命中捞回、范围过滤共用。

为什么「捞回」必须测死：混合检索的全部价值就在这一条——
「关键词强命中但余弦擦边」的块，纯向量模式必须漏、hybrid 模式必须救回来。
这是论文实验表里 hybrid 优于 vector 的证据链起点。
"""
import pytest

from app.core.config import settings
from app.rag import embeddings, hybrid, retriever

TEXT_KEY = "快速傅里叶变换可以将时域信号转换到频域，广泛应用于信号处理。"
TEXT_NOISE = "猫喜欢吃鱼，狗喜欢吃骨头，这是动物习性常识。"


class _KeyedEmbed:
    """确定性假向量：含关键词的文本给一个方向，其余给近乎垂直的方向。

    为什么不用 conftest 的 n-gram 假向量：n-gram 版对共享字敏感，
    「无关文本」也会有中等余弦，构造不出「向量闸必拒、BM25 闸必过」的极端场景。
    async __call__ 与 embed_texts 契约一致（调用方一律 await）。
    """

    async def __call__(self, texts):
        return [[1.0, 0.0] if "快速傅里叶变换" in t else [0.0, 0.99] for t in texts]


# ===== 纯函数 =====


def test_rrf_fuse_consensus_first():
    """两路共识（c 同时在两榜）压过单路第一（RRF 核心语义）。"""
    assert hybrid.rrf_fuse(["a", "b", "c"], ["c", "d"]) == ["c", "a", "b", "d"]


def test_fuse_dual_gate_union():
    """双闸并集：向量过闸 OR BM25 过闸即入；两闸都不过即使双榜有名也被拒。"""
    vector_ranked = ["v1", "v2"]
    bm25_ranked = ["b1", "v2"]
    out = hybrid.fuse(
        vector_ranked,
        bm25_ranked,
        {"v1": 0.5, "v2": 0.2},   # v1 过向量闸；v2 擦边不过
        {"b1": 5.0, "v2": 1.0},   # b1 过 BM25 闸；v2 的 BM25 也不够
        threshold=0.3,
        floor=2.0,
    )
    assert set(out) == {"v1", "b1"}  # v2 两闸都不过 → 拒
    assert out[0] == "v1"  # 合法候选按 RRF 序（v1 首现于前）


def test_tokenize_mixed_language():
    """中英混排：中文走 jieba、英文数字小写化（FFT→fft 可匹配）。"""
    tokens = hybrid.tokenize("用 FFT 计算 DFT，复杂度 O(n log n)")
    assert "fft" in tokens
    assert any("计算" in t or "复杂" in t for t in tokens)


def test_cosine_basic():
    assert hybrid.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert hybrid.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert hybrid.cosine([], [1.0]) == 0.0


# ===== 集成：混合检索对向量闸的「捞回」 =====


async def _seed_store(store):
    await store.upsert(
        ids=["1:0", "1:1"],
        texts=[TEXT_KEY, TEXT_NOISE],
        metadatas=[
            {"file_name": "信号处理.pdf", "page_no": 7, "chunk_index": 0, "doc_id": 1, "group_id": 1},
            {"file_name": "杂记.docx", "page_no": 1, "chunk_index": 1, "doc_id": 1, "group_id": 1},
        ],
    )


@pytest.mark.asyncio
async def test_hybrid_rescues_keyword_hit_vector_rejects(monkeypatch, make_store):
    """★核心用例：阈值高到向量闸全灭（1.1>1 不可能通过），BM25 关键词闸把术语块救回来。"""
    keyed = _KeyedEmbed()
    monkeypatch.setattr(embeddings, "embed_texts", keyed)
    store = make_store()
    await _seed_store(store)

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "score_threshold", 1.1)  # 向量闸物理全灭
    monkeypatch.setattr(settings, "bm25_floor", 1.0)

    hits = await retriever.retrieve("快速傅里叶变换", user_id=1, group_ids=[1], stores={1: store})
    assert hits, "BM25 必须捞回关键词命中（混合检索的全部价值所在）"
    assert hits[0].file_name == "信号处理.pdf"

    # 对照：同条件纯向量模式必须空手（证明差异来自混合路径而不是运气）
    monkeypatch.setattr(settings, "retrieval_mode", "vector")
    hits_vec = await retriever.retrieve("快速傅里叶变换", user_id=1, group_ids=[1], stores={1: store})
    assert hits_vec == []


@pytest.mark.asyncio
async def test_hybrid_respects_scope_filter(monkeypatch, make_store):
    """混合路径共用范围过滤：页码区间外的关键词命中也不许进来。"""
    monkeypatch.setattr(embeddings, "embed_texts", _KeyedEmbed())
    store = make_store()
    await _seed_store(store)
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "score_threshold", 1.1)  # 逼纯 BM25 路
    monkeypatch.setattr(settings, "bm25_floor", 1.0)

    hits = await retriever.retrieve(
        "快速傅里叶变换", user_id=1, group_ids=[1], stores={1: store}, page_range=(1, 3)
    )
    assert hits == []  # 关键词块在第 7 页，被范围过滤挡掉


@pytest.mark.asyncio
async def test_vector_mode_never_touches_bm25(monkeypatch, make_store):
    """vector 模式不碰混合检索代码：get_all 一被调用即测试失败（零多余开销承诺）。"""
    monkeypatch.setattr(embeddings, "embed_texts", _KeyedEmbed())
    store = make_store()
    await _seed_store(store)
    monkeypatch.setattr(settings, "retrieval_mode", "vector")

    async def _forbidden(self):
        raise AssertionError("vector 模式严禁调用 get_all（BM25 语料拉取）")

    monkeypatch.setattr(type(store), "get_all", _forbidden)
    hits = await retriever.retrieve("快速傅里叶变换", user_id=1, group_ids=[1], stores={1: store})
    assert hits  # 纯向量路径正常工作（阈值默认，keyed 向量余弦=1 命中）
