# -*- coding: utf-8 -*-
"""
pytest 全局 fixture（全项目复用的「三大件」：数据库 / 向量库 / 假向量化）。

为什么在 conftest 里集中造 fixture：
1. 每个测试拿到干净的 SQLite 内存库——零外部依赖、不污染真实 MySQL、跑完即弃；
2. M2 起向量库（chroma_client）与假向量（fake_embed_fn）也在这里加 fixture，
   测试写法统一；单测零网络、零磁盘、可重复（显存红线：单测绝不打真实 Ollama）。
"""
import hashlib
import itertools
import uuid

import pytest
from chromadb import EphemeralClient
from sqlalchemy.orm import Session

from app.db.session import Base, make_engine
from app.rag.vector_store import ChromaStore


@pytest.fixture(autouse=True)
def _pin_retrieval_mode_vector(monkeypatch):
    """测试默认锁「纯向量」模式——测试结果不得随开发机 .env 漂移。

    2026-09-25 实例：产品经理在 .env 开了 RETRIEVAL_MODE=hybrid 后，
    全量 pytest 跟着跑进 hybrid 路径，既有向量用例的语义悄悄变了。
    默认钉死 vector 保证基线可复现；测混合检索的用例在自己体内
    monkeypatch 覆盖为 hybrid（测试内 setattr 晚于本夹具，必然生效）。
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "retrieval_mode", "vector")


@pytest.fixture()
def db_session():
    """干净的 SQLite 内存会话：每个测试独立建表，互不干扰。

    为什么用 make_engine("sqlite:///:memory:")：StaticPool 保证同一连接共享内存库
    （见 app/db/session.py 的注释）。
    """
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture()
def gateway():
    """独立的 OllamaGateway 实例（不共用全局单例的锁状态，测试互不影响）。"""
    from app.core.llm import OllamaGateway

    return OllamaGateway(base_url="http://testserver")


@pytest.fixture()
def chroma_client():
    """每测试一个 EphemeralClient（纯内存向量库，测试间零共享）。

    为什么必须 EphemeralClient 而不是 PersistentClient：
    1. 落盘客户端会写开发机的 settings.chroma_dir 真实数据，污染演示数据；
    2. 落盘数据让断言依赖测试执行顺序，重复跑结果不一致；
    3. 内存客户端随 fixture 创建/销毁，天然按测试隔离（这也是 vector_store.py
       里「单测显式传 EphemeralClient，完全不走持久化路径」约定的落点）。
    """
    return EphemeralClient()


# 假向量固定维度：32 维足以拉开「相似 / 无关」文本的余弦差距，测试断言也好写
_FAKE_EMBED_DIM = 32


class _AwaitableVectors(list):
    """既能当普通 list 同步用、又能被 await 的向量列表（测试缝的兼容层）。

    为什么需要双形态：fake_embed_fn 的契约签名是同步的 (texts)->list[list[float]]
    （便于测试直接拿期望向量做断言），但 monkeypatch 到 app.rag.embeddings.embed_texts
    之后，retrieve / ChromaStore.upsert 一律 `await embed_texts(...)`——
    若返回裸 list，await 会 TypeError。给列表补一个立即结束的 __await__，
    同步取值与 await 取值两种用法就都成立。
    """

    def __await__(self):
        # yield from () 只为让本函数成为生成器函数（__await__ 必须返回迭代器）；
        # 生成器立刻结束并交出纯 list——不要把子类实例漏给 chroma 当 embeddings 用。
        yield from ()
        return list(self)


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """确定性假向量核心：字符 n-gram + 带符号哈希 → 固定维度 → L2 归一化。

    为什么这样设计（而不是内置 hash() 或随机向量）：
    1. 内置 hash(str) 受 PYTHONHASHSEED 随机化影响，跨进程不可复现，测试会飘；
    2. 随机向量保证不了「相似文本余弦相近、无关文本远离」，检索相关性断言没法写；
    3. 带符号哈希（同一 n-gram 恒定落入同一维、且恒定取同一符号）让无关文本
       各维正负相抵、余弦趋近 0，而共享 n-gram 多的文本方向一致、余弦趋近 1，
       正好覆盖检索相关性断言；
    4. L2 归一化后余弦相似度 = 点积，断言只看 [-1, 1] 一个数。
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _FAKE_EMBED_DIM
        lowered = (text or "").lower()
        # 1/2/3-gram 都计入：短句靠 1-gram 撑方向，长句靠 2/3-gram 拉开区分度；
        # 权重随 n 递增——越长的 n-gram 越具体，越该主导向量方向
        for n in (1, 2, 3):
            for i in range(len(lowered) - n + 1):
                gram = lowered[i : i + n]
                # 用 sha256 而不是内置 hash：跨进程稳定；digest 拆成「桶 + 符号」
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % _FAKE_EMBED_DIM
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign * float(n)
        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0.0:
            # 空文本兜底成固定单位向量：既避免除零，同类空文本之间余弦也恒为 1
            vec[0] = 1.0
        else:
            vec = [x / norm for x in vec]
        vectors.append(vec)
    # 必须包一层 _AwaitableVectors：调用方一律 `await embeddings.embed_texts(...)`，
    # 裸 list 会 TypeError（上面的双形态设计就是为这里服务的）
    return _AwaitableVectors(vectors)


@pytest.fixture()
def fake_embed_fn():
    """确定性假向量函数，签名 (texts: list[str]) -> list[list[float]]。

    供 monkeypatch app.rag.embeddings.embed_texts / 直接传入 retrieve 的测试缝：
    单测绝不打真实 Ollama（显存红线 + 答辩机断网风险），用字符 n-gram 哈希假向量
    保证相似文本向量余弦相近、无关文本远离，检索相关性断言可离线成立。

    用法示例（返回值可同步当 list 用，也可 await）：
    - monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    - vecs = fake_embed_fn(["知识库"])  # 同步取期望向量做断言
    """
    return _fake_embed_fn


@pytest.fixture()
def make_store(chroma_client):
    """ChromaStore 工厂：每次造一个独占 collection 的向量库实例。

    为什么 collection 名要加序号+随机后缀：get_or_create_collection 同名会复用
    旧数据，同一测试里连建两个 store 就会互踩断言；后缀保证物理隔离。
    为什么不直接用 store_for：store_for 的命名 u{user_id}g{group_id} 是层间契约，
    只有专测 store_for 命名时才该用它；本工厂服务「随手造个库」的场景，
    名字刻意避开契约命名，防止测试数据混进契约 collection 规则里被误断言。
    """
    seq = itertools.count()

    def _make(name: str | None = None) -> ChromaStore:
        """建一个绑定独立 collection 的 ChromaStore（缺省名带序号+随机后缀）。"""
        col_name = name if name is not None else f"test{next(seq)}_{uuid.uuid4().hex[:8]}"
        # 显式传入 chroma_client（EphemeralClient）：绝不落盘，红线同 vector_store 注释
        return ChromaStore(collection=col_name, client=chroma_client)

    return _make
