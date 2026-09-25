# -*- coding: utf-8 -*-
"""
Ollama 调用网关：★ 显存红线（CLAUDE.md §3）的代码落点，全项目命门。

为什么要有这一层封装（而不是各处散落 httpx.post）：
1. 8G 显存下必须保证「任一时刻只有一类大模型（LLM/OCR）活跃」——
   所有生成类调用都必须过「排队闸门 + 互斥锁」，散落调用无法统一约束，
   一个并发上传 + 并发提问就可能把显存打爆；
2. keep_alive 驻留策略集中在这里（LLM 热点常驻 / OCR 用完即卸），业务代码不关心显存细节；
3. 生产升级路径（README 对比表第 1 行）：把本地锁换成 Redis 分布式锁、
   把 HTTP 调用换成推理服务 RPC，只需要改这一个文件。

锁的设计说明（为什么是「信号量 + 锁」两个原语）：
- `Semaphore(1)` 是「排队闸门」：并发请求在这里排队，同一时刻只放行 1 个——
  语义是队列宽度，将来若允许 embedding 与生成并行，可单独再开闸；
- `asyncio.Lock` 是「模型互斥锁」：表达「同一时刻只有一类大模型在飞」的红线语义，
  M5 的 OCR「先卸 LLM 再跑 OCR」序列会围绕它做临界区；接口按队列语义写，
  生产环境把实现换成 Redis 分布式锁即可。
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.core.config import settings
from app.core.exceptions import UpstreamError

logger = logging.getLogger(__name__)

# 显存红线：embedding 批量限流——一次最多 16 条（0.6B 小模型，可与 LLM 共存，但批量要有界）
EMBED_BATCH_SIZE = 16


class OllamaGateway:
    """Ollama HTTP 封装：生成 / 向量化 / 卸载 / 健康检查。"""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout  # 生成默认 300s：7B 首次冷加载可能要几十秒
        # 排队闸门 + 模型互斥锁（见模块 docstring 的设计说明）
        self._queue = asyncio.Semaphore(1)
        self._mutex = asyncio.Lock()

    # ---------- 基础 HTTP（测试可 monkeypatch 这两个方法，零网络跑单测） ----------

    async def _post_json(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        """POST JSON 到 Ollama，网络/上游错误统一转 UpstreamError（人话提示）。"""
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as client:
                resp = await client.post(path, json=payload, timeout=timeout or self.timeout)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(f"Ollama 返回错误状态 {e.response.status_code}") from e
        except httpx.HTTPError as e:
            # 连接失败/超时：典型场景是 Ollama 服务没启动
            raise UpstreamError("无法连接 Ollama 服务（是否已启动？）") from e

    async def _get_json(self, path: str, timeout: float | None = None) -> dict:
        """GET JSON from Ollama（健康检查用）。"""
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as client:
                resp = await client.get(path, timeout=timeout or 10.0)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            raise UpstreamError("无法连接 Ollama 服务（是否已启动？）") from e

    # ---------- 生成类调用（LLM / OCR 共用，必须串行） ----------

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        keep_alive: str | int | None = None,
        num_ctx: int | None = None,
        num_predict: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """一次生成调用（文本或带图）。返回生成文本。

        参数说明：
        - images: base64 图片列表（视觉/OCR 用，M5 起传入——GLM-OCR 就是走这里）；
        - keep_alive: 模型用完后驻留多久。OCR 传 0 =「用完即卸」（显存红线）；
        - num_predict: 限制输出长度（红线：输出长度受控），默认 512 token；
        - temperature: 意图分类等确定性任务由调用方传 0。
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # 未指定时默认用 LLM 的驻留策略（热点常驻，省重复加载时间）
            "keep_alive": settings.llm_keep_alive if keep_alive is None else keep_alive,
            "options": {
                # 上下文上限：红线要求 ≤4096，OCR 单页等场景由调用方覆盖更小值
                "num_ctx": settings.llm_num_ctx if num_ctx is None else num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        # ★ 临界区：排队闸门 + 互斥锁，保证同一时刻只有一个生成请求在飞
        async with self._queue:
            async with self._mutex:
                data = await self._post_json("/api/generate", payload)
        return data.get("response", "")

    async def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        keep_alive: str | int | None = None,
        num_ctx: int | None = None,
        num_predict: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """流式生成：逐段 yield 文本增量（参数语义与 generate 完全一致）。

        为什么锁要覆盖「整段流」而不是每段一把：红线管的是「同一时刻只有一个
        生成在飞」——流式下一次生成横跨数秒/数十秒，若逐段释放锁，另一个请求
        会在两段之间插进来变成并发推理。queue+mutex 从首个字节持有到最后一个
        字节，流式只是传输方式变化，互斥语义与非流式一字不差。

        为什么不用 _post_json：它等整包 JSON；Ollama stream=true 返回的是
        逐行 NDJSON（{"response":"token"}），必须走 httpx.stream 逐行读。
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,  # ★ 流式开关（与 generate 的唯一差异）
            "keep_alive": settings.llm_keep_alive if keep_alive is None else keep_alive,
            "options": {
                "num_ctx": settings.llm_num_ctx if num_ctx is None else num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        async with self._queue:
            async with self._mutex:
                try:
                    async with httpx.AsyncClient(base_url=self.base_url) as client:
                        async with client.stream(
                            "POST", "/api/generate", json=payload, timeout=self.timeout
                        ) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    continue  # 非 JSON 行（偶发日志行）跳过，不让整条流崩掉
                                piece = data.get("response", "")
                                if piece:
                                    yield piece
                                if data.get("done"):
                                    break
                except httpx.HTTPStatusError as e:
                    raise UpstreamError(f"Ollama 返回错误状态 {e.response.status_code}") from e
                except httpx.HTTPError as e:
                    raise UpstreamError("无法连接 Ollama 服务（是否已启动？）") from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化，返回与输入等长的向量列表。

        为什么分批：红线要求 embedding 批量限流（一次 ≤16 条），
        一次几百条会把请求拖死、显存/内存也会抖。
        """
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """单批向量化。优先用批量接口 /api/embed，若版本不支持则回退逐条老接口。"""
        try:
            data = await self._post_json(
                "/api/embed",
                {"model": settings.embed_model, "input": batch},
                timeout=60.0,
            )
            return list(data.get("embeddings") or [])
        except UpstreamError:
            # 回退：/api/embeddings 老接口逐条调用（兼容不同 Ollama 版本）
            vectors = []
            for text in batch:
                data = await self._post_json(
                    "/api/embeddings",
                    {"model": settings.embed_model, "prompt": text},
                    timeout=60.0,
                )
                vectors.append(list(data.get("embedding") or []))
            return vectors

    async def unload(self, model: str) -> None:
        """立即卸载模型，释放显存（显存红线：OCR 用完即卸的实现）。

        Ollama 的标准姿势：发一次 keep_alive=0 的空生成 = 跑完立刻从显存踢掉。
        M5 的「入库前先卸 LLM」就是调这个方法。
        """
        async with self._mutex:
            await self._post_json(
                "/api/generate",
                {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=60.0,
            )

    async def active_models(self) -> list[str]:
        """查询当前驻留在显存中的模型名列表（Ollama /api/ps）。

        为什么需要它：M3 的「OCR 前先卸 LLM」必须先确认 LLM 是否真的驻留——
        对未加载的模型发 keep_alive=0 空生成，Ollama 会先加载再立刻卸载（白等几十秒）；
        查一下再决定卸不卸，互斥序列既安全又不浪费。
        这也是单测的缝：测试里 monkeypatch 本方法即可模拟「驻留/未驻留」两种时序。
        """
        data = await self._get_json("/api/ps")
        return [m.get("name", "") for m in data.get("models", [])]

    async def health(self) -> dict:
        """健康检查：Ollama 是否连通 + 三个配置模型是否都在列。

        返回 {"ok": bool, "models": [...], "missing": [...]}，
        /health 接口直接透出——演示/部署时一眼看出环境问题（就像昨天查网络那样）。
        """
        try:
            data = await self._get_json("/api/tags")
        except UpstreamError as e:
            return {"ok": False, "models": [], "missing": [], "error": e.message}

        tags = [m.get("name", "") for m in data.get("models", [])]
        wanted = [settings.llm_model, settings.ocr_model, settings.embed_model]
        missing = [w for w in wanted if not self._match_tag(tags, w)]
        return {"ok": True, "models": tags, "missing": missing}

    @staticmethod
    def _match_tag(tags: list[str], want: str) -> bool:
        """模型名匹配：`glm-ocr` 能匹配 `glm-ocr` 或 `glm-ocr:latest`（标签可省略）。"""
        return any(t == want or t.startswith(want + ":") for t in tags)


# 进程级单例：全项目共用同一把锁（多把锁 = 红线失效）
gateway = OllamaGateway()
