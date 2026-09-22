# -*- coding: utf-8 -*-
"""
M1 单测：★ 推理互斥锁（显存红线的自动化守卫）。

为什么必须测锁本身：红线「任一时刻只有一类大模型活跃」若只靠人工 review，
迟早被某个并发上传/并发提问打破——用并发测试把「串行化」钉死在 CI 里。
测试通过 monkeypatch 掉真实 HTTP（_post_json），零网络零 GPU，毫秒级跑完。
"""
import asyncio

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_concurrent_generate_serialized(gateway, monkeypatch):
    """5 个并发 generate 必须串行执行（同一时刻在飞的请求永远 ≤1）。"""
    active = 0
    max_active = 0

    async def fake_post(path, payload, timeout=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)  # 模拟推理耗时，给并发交错的机会
        active -= 1
        return {"response": "ok"}

    monkeypatch.setattr(gateway, "_post_json", fake_post)
    await asyncio.gather(*[gateway.generate(model="m", prompt=f"p{i}") for i in range(5)])

    assert max_active == 1, f"互斥锁失效！同一时刻最多有 {max_active} 个推理在飞（红线要求 1）"


@pytest.mark.asyncio
async def test_keep_alive_default_and_override(gateway, monkeypatch):
    """keep_alive 必须正确传给 Ollama：默认取配置（LLM 常驻），可覆盖为 0（OCR 用完即卸）。"""
    captured = {}

    async def fake_post(path, payload, timeout=None):
        captured.clear()
        captured.update(payload)
        return {"response": "x"}

    monkeypatch.setattr(gateway, "_post_json", fake_post)

    await gateway.generate(model="m", prompt="p")
    assert captured["keep_alive"] == settings.llm_keep_alive, "默认应使用 LLM 驻留策略"

    await gateway.generate(model="m", prompt="p", keep_alive=0)
    assert captured["keep_alive"] == 0, "OCR 场景必须能覆盖成用完即卸"


@pytest.mark.asyncio
async def test_generate_limits_output_and_ctx(gateway, monkeypatch):
    """红线：输出长度受控（num_predict 有界）、上下文上限来自配置。"""
    captured = {}

    async def fake_post(path, payload, timeout=None):
        captured.clear()
        captured.update(payload)
        return {"response": "x"}

    monkeypatch.setattr(gateway, "_post_json", fake_post)
    await gateway.generate(model="m", prompt="p")
    assert captured["options"]["num_predict"] <= 512
    assert captured["options"]["num_ctx"] <= 4096  # 红线：LLM_NUM_CTX ≤ 4096


@pytest.mark.asyncio
async def test_embed_batch_limited(gateway, monkeypatch):
    """红线：embedding 批量限流——单次请求最多 EMBED_BATCH_SIZE(16) 条。"""
    from app.core.llm import EMBED_BATCH_SIZE

    batch_sizes = []

    async def fake_post(path, payload, timeout=None):
        batch = payload.get("input") or []
        batch_sizes.append(len(batch))
        return {"embeddings": [[0.1] * 4 for _ in batch]}

    monkeypatch.setattr(gateway, "_post_json", fake_post)
    texts = [f"t{i}" for i in range(40)]
    vectors = await gateway.embed_texts(texts)

    assert len(vectors) == 40
    assert all(n <= EMBED_BATCH_SIZE for n in batch_sizes), f"批量超限: {batch_sizes}"


@pytest.mark.asyncio
async def test_unload_uses_keep_alive_zero(gateway, monkeypatch):
    """卸载模型必须发 keep_alive=0（显存红线：用完即卸的实现正确性）。"""
    captured = {}

    async def fake_post(path, payload, timeout=None):
        captured.clear()
        captured.update(payload)
        return {"response": ""}

    monkeypatch.setattr(gateway, "_post_json", fake_post)
    await gateway.unload("glm-ocr")
    assert captured["keep_alive"] == 0
