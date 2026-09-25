# -*- coding: utf-8 -*-
"""
混合检索（RAG 质量包）：BM25 关键词匹配 + 向量语义检索的融合层（纯 CPU 计算）。

为什么需要 BM25（纯向量的短板）：
qwen3 向量对「中文术语/公式符号/专有名词」的字面不敏感——
「快速傅里叶变换」和它的向量近邻可能语义沾边但不是同一术语；
BM25 按词频精确匹配，正好互补。两路各有过闸标准（双闸）：
- 向量闸：余弦 ≥ SCORE_THRESHOLD（既有防幻觉闸，语义不达标不放行）；
- BM25 闸：BM25 分 ≥ BM25_FLOOR（关键词强命中可独立放行）；
任一闸放行即进入候选，最终顺序由 RRF（倒数排名融合）决定——
RRF 只看名次不看分数量纲，余弦(0~1)与 BM25(十位级)天生不可加，
这是选 RRF 而不是「加权求和」的原因（教科书结论，k=60）。

为什么全部是同步纯函数：本模块零网络零磁盘零模型调用——
I/O（向量查询/语料拉取）留在 retriever，这里只做计算，
单测可以脱离 Chroma/Ollama 直接断言融合与准入行为。
"""
import math
import re

# RRF 常数（教科书默认）：k=60 对「头部排名」与「全榜参与」的平衡最好
_RRF_K = 60

# BM25 候选/向量候选的放大量：融合前各路多取一倍，给对方「捞回」的空间
_CANDIDATE_MULTIPLIER = 2

# 英文/数字词元：jieba 对英文是整词切分，这里额外按词边界拆开，
# 让 "FFT" / "Dijkstra" 这类术语在大小写不敏感下可匹配
_ASCII_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文 jieba 分词 + 英文小写词元（BM25 的匹配粒度在此定义）。

    懒导入 jieba：首次 import 会加载字典（约 1 秒），纯向量模式根本不该付这笔成本——
    只有 hybrid 路径走到这里才初始化。
    """
    import jieba

    lowered = (text or "").lower()
    tokens = [t for t in jieba.lcut(lowered) if t.strip()]
    tokens.extend(_ASCII_TOKEN.findall(lowered))
    return tokens


def build_bm25(texts: list[str]):
    """按语料建 BM25 索引（平滑 IDF）；空语料返回 None（调用方退化为纯向量）。

    为什么不用 rank_bm25 默认 IDF（单测实测踩坑，2026-09-25）：
    Okapi 公式 idf=ln((N-df+0.5)/(df+0.5)) 在**小语料**下会算出 0 甚至负数——
    实测 2 篇文档、术语出现在 1 篇 → ln(1.5/1.5)=0 分，术语明明在文档里却
    一个字都匹配不上；本项目用户分组常只有几份课件，必踩。
    rank_bm25 只给「负数」设了 epsilon 地板、0 分原样放行，所以这里整体换成
    恒正的平滑公式 idf=ln(1+(N-df+0.5)/(df+0.5))：
    小语料恒正、大语料逼近标准公式（N=1000,df=1 时两者均 ≈6.5），排序语义不变。
    """
    if not texts:
        return None
    from collections import Counter

    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(t) for t in texts]
    index = BM25Okapi(tokenized)
    # 文档频率按「出现过该词的文档数」计（同篇重复词只算一次）
    df = Counter()
    for doc in tokenized:
        df.update(set(doc))
    corpus_size = len(tokenized)
    for word, doc_freq in list(index.idf.items()):
        d = max(1, df.get(word, 1))
        index.idf[word] = math.log(1.0 + (corpus_size - d + 0.5) / (d + 0.5))
    return index


def rrf_fuse(*ranked_lists: list[str], k: int = _RRF_K) -> list[str]:
    """RRF 融合：score(id) = Σ 1/(k+rank_i)，按总分降序返回 id 列表。

    同一名次在两路都命中 → 双份加分（两路共识排前）；
    只在一路出现 → 单份分。排名并列时按首次出现顺序稳定排序（可复现断言）。
    """
    scores: dict[str, float] = {}
    order: list[str] = []  # 记录首次出现顺序，并列时作稳定 tie-break
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            if item not in scores:
                order.append(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(order, key=lambda i: (-scores[i], order.index(i)))


def fuse(
    vector_ranked: list[str],
    bm25_ranked: list[str],
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    *,
    threshold: float,
    floor: float,
) -> list[str]:
    """双闸准入 + RRF 排序，返回最终候选 id 序（调用方再截 top_k）。

    准入是并集语义：向量闸放行 ∪ BM25 闸放行——
    「语义够但字面没有」与「字面强命中但余弦擦边」两类都救得回来；
    两闸都不过的垃圾两路互不背书，依然进不来（防幻觉口径不降级）。
    """
    admitted = {i for i in vector_ranked if vector_scores.get(i, -1.0) >= threshold}
    admitted |= {i for i in bm25_ranked if bm25_scores.get(i, -1.0) >= floor}
    order = rrf_fuse(vector_ranked, bm25_ranked)
    return [i for i in order if i in admitted]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（BM25 独家命中的存量向量补算分数用）。

    手算而不是调库：维度只有几百、单次调用，不值得为它引 numpy 之外的依赖
    （chroma 内部的余弦与这里必须同口径——都是归一化后点积）。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
