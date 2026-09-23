# -*- coding: utf-8 -*-
"""
文本向量化薄封装：写路径（入库）与读路径（提问）共用的唯一向量化入口。

为什么单独留这一层（而不是各处直接敲 gateway.embed_texts）：
1. ★ 这是单测 monkeypatch 的缝——单测绝不打真实 Ollama，
   统一 patch `app.rag.embeddings.embed_texts` 这一个点，
   入库向量化、提问向量化就全部换成确定性假向量，测试零网络可跑；
2. 批量限流（一次 16 条）与重试/回退策略都封装在 gateway 内，
   这里只负责「接线」，将来换 embedding 服务（如生产环境的推理服务 RPC）
   也只动 gateway 或本文件，上层零改动。

调用约定（保证 monkeypatch 一处生效）：
调用方必须写 `from app.rag import embeddings` 然后 `await embeddings.embed_texts(...)`，
不要 `from app.rag.embeddings import embed_texts`——后者会在导入时把函数对象绑定走，
patch 原模块后调用方仍指向旧函数，缝就失效了。
"""
from app.core.llm import gateway


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本向量化，返回与输入等长的向量列表。

    为什么必须经 gateway：显存红线（CLAUDE.md §3）要求一切 Ollama 调用走统一网关
    （批量限流已封装在 gateway.embed_texts 内），绕过网关直连 HTTP 会破坏限流约束。

    ★ 这是单测 monkeypatch 的缝：tests 里 patch 本函数即可离线跑通整条检索链路。
    """
    return await gateway.embed_texts(texts)
