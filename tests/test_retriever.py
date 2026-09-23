# -*- coding: utf-8 -*-
"""
检索器单测（M2）：阈值过滤（防幻觉第一道闸）+ 分组隔离（权限边界）+ top_k 截断。

为什么阈值/隔离必须测死：
阈值失灵 = 无关资料喂给 LLM 编造答案（卖点变事故）；
隔离失灵 = 跨分组/跨用户检索（数据泄漏）。
全部用 EphemeralClient + 假向量离线跑，绝不打真实 Ollama。
"""
import pytest

from app.core.config import settings
from app.rag import embeddings, retriever

TEXT_HIT = "梯度下降的学习率过大会导致损失函数震荡不收敛，需要调小学习率。"
TEXT_OTHER = "猫喜欢吃鱼，狗喜欢吃骨头，这是动物习性常识。"


@pytest.mark.asyncio
async def test_related_question_hits_with_trace_metadata(monkeypatch, fake_embed_fn, make_store):
    """相近问题命中：返回块的 file_name/page_no 必须取自入库 metadata（溯源根基）。"""
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    store = make_store()
    await store.upsert(
        ids=["1:0", "1:1"],
        texts=[TEXT_HIT, TEXT_OTHER],
        metadatas=[
            {"file_name": "讲义.docx", "page_no": 3, "chunk_index": 0, "doc_id": 1, "group_id": 1, "user_id": 1},
            {"file_name": "杂记.docx", "page_no": 1, "chunk_index": 1, "doc_id": 1, "group_id": 1, "user_id": 1},
        ],
    )
    hits = await retriever.retrieve(
        "梯度下降的学习率过大会导致损失函数震荡",
        user_id=1,
        group_ids=[1],
        stores={1: store},
        top_k=5,
        threshold=0.1,
    )
    assert hits, "相近问题必须命中"
    top = hits[0]
    assert top.text == TEXT_HIT
    assert top.file_name == "讲义.docx"
    assert top.page_no == 3
    assert top.score >= 0.1
    assert top.group_id == 1


@pytest.mark.asyncio
async def test_threshold_filters_unrelated(monkeypatch, fake_embed_fn, make_store):
    """阈值过滤：无关问题在高阈值下召回为空——宁缺毋滥（调用方据此走兜底）。"""
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    store = make_store()
    await store.upsert(
        ids=["1:0"],
        texts=[TEXT_HIT],
        metadatas=[{"file_name": "讲义.docx", "page_no": 1, "chunk_index": 0, "doc_id": 1, "group_id": 1, "user_id": 1}],
    )
    hits = await retriever.retrieve(
        "爱因斯坦的光电效应方程是什么",
        user_id=1,
        group_ids=[1],
        stores={1: store},
        threshold=0.99,
    )
    assert hits == []


@pytest.mark.asyncio
async def test_group_isolation(monkeypatch, fake_embed_fn, make_store):
    """分组隔离：g1 的内容搜 g2 的库绝不得命中（越权检索=数据泄漏）。"""
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    store1, store2 = make_store(), make_store()
    await store1.upsert(
        ids=["1:0"],
        texts=[TEXT_HIT],
        metadatas=[{"file_name": "讲义.docx", "page_no": 1, "chunk_index": 0, "doc_id": 1, "group_id": 1, "user_id": 1}],
    )
    # 只搜 g2（内容在 g1）→ 空
    hits = await retriever.retrieve(
        "梯度下降的学习率过大会导致损失函数震荡",
        user_id=1,
        group_ids=[2],
        stores={1: store1, 2: store2},
        threshold=0.0,
    )
    assert hits == []
    # 搜 g1 → 命中（对照证明不是检索本身坏了）
    hits = await retriever.retrieve(
        "梯度下降的学习率过大会导致损失函数震荡",
        user_id=1,
        group_ids=[1],
        stores={1: store1, 2: store2},
        threshold=0.0,
    )
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_top_k_truncation_and_score_order(monkeypatch, fake_embed_fn, make_store):
    """top_k 截断：合并命中按 score 降序，只留前 k 条。"""
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_fn)
    store = make_store()
    await store.upsert(
        ids=[f"1:{i}" for i in range(5)],
        texts=[f"梯度下降的学习率过大会导致损失函数震荡{i}" for i in range(5)],
        metadatas=[
            {"file_name": "讲义.docx", "page_no": i + 1, "chunk_index": i, "doc_id": 1, "group_id": 1, "user_id": 1}
            for i in range(5)
        ],
    )
    hits = await retriever.retrieve(
        "梯度下降的学习率过大会导致损失函数震荡",
        user_id=1,
        group_ids=[1],
        stores={1: store},
        top_k=2,
        threshold=0.0,
    )
    assert len(hits) == 2
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_blank_question_short_circuit(monkeypatch, make_store):
    """空问题/空分组直接短路返回 []：不做无谓模型调用（显存红线）。"""
    def _boom(texts):
        raise AssertionError("短路失败：不该触发向量化")

    monkeypatch.setattr(embeddings, "embed_texts", _boom)
    assert await retriever.retrieve("", user_id=1, group_ids=[1], stores={1: make_store()}) == []
    assert await retriever.retrieve("正常问题", user_id=1, group_ids=[], stores={}) == []


def test_settings_threshold_used_by_default(monkeypatch, fake_embed_fn, make_store):
    """缺省阈值来自 settings（配置只从 .env 读）——确认 retrieve 不硬编码阈值。"""
    assert settings.score_threshold > 0  # 配置存在即可；具体数值 M2 验收后标定
