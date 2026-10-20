# -*- coding: utf-8 -*-
"""
服务化网关 v1（M7-2）：把「知识库 + RAG + Agent」以 API Key 鉴权的稳定契约
提供给其他项目调用。

为什么要独立一层 /v1 而不是让外部直接调 /chat、/kb：
1. 鉴权方式不同：外部是机器调用（X-API-Key），内部前端是人机登录（JWT）；
   两套入口分开，避免在既有接口里写「要么 Bearer 要么 API Key」的分支；
2. 契约可独立演进：/v1 的字段与错误码面向调用方文档化，内部接口改动
   不会无声破坏外部集成（版本前缀就是这条边界）；
3. 复用而不是复制：分组校验、问答上下文、入库流水线全部调用既有函数
   （chat.resolve_group_ids / chat.prepare_context / kb.ingest_uploads），
   业务只有一份实现，避免 M6 改写污染那样的双路径漂移。

容量与背压（产品经理 2026-10-20 确认的物理前提）：
8G 显存 + 单机 Ollama ⇒ 同一时刻只能有一个推理请求（gateway 的 Semaphore(1)）。
本层再加一道容量闸门，排队超过 V1_QUEUE_TIMEOUT 就拒绝：
- JSON 端点：HTTP 429 + Retry-After；
- SSE 端点：流内 error 事件（code=429）——流一旦开始，HTTP 状态码已发出。
这不是额外的并发能力，而是把「排队会很久」提前、明确地告诉调用方。
"""
import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, Header, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.api.chat import prepare_context, resolve_group_ids
from app.api.kb import ingest_uploads
from app.core.config import settings
from app.core.exceptions import AuthError, NotFoundError, TooManyRequestsError
from app.core.security import hash_api_key
from app.db import crud
from app.db.models import User, split_empty_pages
from app.db.session import get_db
from app.rag.retriever import retrieve

router = APIRouter(prefix="/v1", tags=["v1"])
logger = logging.getLogger(__name__)

# 容量闸门：固定 1，与 gateway 的推理串行锁同源。
# 为什么不做成可配：调大并发 = 多个请求同时打 Ollama，8G 显存直接爆——
# 红线（CLAUDE.md §3）不允许「配置能开出来的危险」。
_v1_gate = asyncio.Semaphore(1)


def get_current_app_user(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    db=Depends(get_db),
) -> User:
    """API Key 鉴权依赖：X-API-Key → 哈希查表 → 账号。

    为什么直接返回 User：一个 Key 绑定一个系统账号（见 ApiKey 模型注释），
    后续所有端点继续用 user.id 做数据隔离——与 JWT 登录路径完全同构。

    安全口径：Key 缺失/无效/已吊销统一 401 同一提示，不区分原因
    （与登录接口「用户名或密码错误」同款防枚举纪律）。
    """
    if not x_api_key:
        raise AuthError("请携带 X-API-Key 请求头")
    key = crud.get_active_api_key(db, key_hash=hash_api_key(x_api_key))
    if key is None:
        raise AuthError("API Key 无效或已吊销")
    user = crud.get_user_by_id(db, key.user_id)
    if user is None:
        raise AuthError("API Key 关联的账号不存在")
    crud.touch_api_key(db, key)
    return user


async def _acquire_gate() -> None:
    """等待容量闸门，超时抛 429（JSON 端点用）。

    为什么用 wait_for 而不是无限 await：调用方等 30 秒还没轮到，
    不如立刻拿到「稍后重试」的明确信号——排队不是错误，但无限挂起是。
    """
    try:
        await asyncio.wait_for(_v1_gate.acquire(), timeout=settings.v1_queue_timeout)
    except asyncio.TimeoutError:
        raise TooManyRequestsError(
            "服务繁忙，请稍后重试", headers={"Retry-After": "5"}
        )


async def _run_with_gate(factory):
    """在容量闸门内执行协程工厂（工厂形式避免超时后残留未 await 的协程）。

    注意：闸门只保证「同一时刻一个 /v1 请求进入」，真正的显存串行仍由
    gateway 的锁保证（内部前端与 /v1 共享同一把锁）。
    """
    await _acquire_gate()
    try:
        return await factory()
    finally:
        _v1_gate.release()


def _sse(obj: dict) -> str:
    """dict → SSE data 行（与 /chat/ask 流式同一事件契约）。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ===== 请求模型 =====


class RetrieveRequest(BaseModel):
    """纯检索入参：调用方自己拼 prompt 时用（不触发 LLM 生成）。"""

    query: str = Field(min_length=1, max_length=2000)
    group_ids: list[int] | None = None  # None=账号下全部分组
    # 上限 20：防止单次检索把内存/模型批量拖垮（服务化必须假设调用方会乱传）
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequestV1(BaseModel):
    """问答入参（无状态单轮；多轮由调用方自行维护上下文）。"""

    question: str = Field(min_length=1, max_length=2000)
    group_ids: list[int] | None = None
    guide_mode: bool = False


class GroupCreateV1(BaseModel):
    name: str = Field(min_length=1, max_length=128)


# ===== 知识库管理 =====


@router.get("/kb/groups")
def list_groups(user: User = Depends(get_current_app_user), db=Depends(get_db)):
    """本人账号下的知识分组（调用方先用它拿 group_id）。"""
    groups = crud.list_kb_groups(db, user_id=user.id)
    return [
        {
            "id": g.id,
            "name": g.name,
            "created_at": g.created_at,
            # 分组数是个人知识库量级，逐组统计可接受（与 /kb/groups 同口径）
            "doc_count": len(crud.list_documents(db, user_id=user.id, group_id=g.id)),
        }
        for g in groups
    ]


@router.post("/kb/groups", status_code=201)
def create_group(
    body: GroupCreateV1,
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """建知识分组（同名 409，由 crud 统一翻译）。"""
    group = crud.create_kb_group(db, user_id=user.id, name=body.name)
    return {"id": group.id, "name": group.name}


@router.get("/kb/groups/{gid}/documents")
def list_documents(
    gid: int,
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """分组下文档清单（非本人分组 404 防探测）。"""
    if crud.get_kb_group(db, user_id=user.id, group_id=gid) is None:
        raise NotFoundError("分组不存在")
    docs = crud.list_documents(db, user_id=user.id, group_id=gid)
    return [
        {
            "id": d.id,
            "file_name": d.file_name,
            "status": d.status,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "empty_pages": split_empty_pages(d.empty_pages),
            "updated_at": d.updated_at,
        }
        for d in docs
    ]


@router.post("/kb/groups/{gid}/documents")
async def upload_documents(
    gid: int,
    file: list[UploadFile] = File(...),
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """批量上传入库：与内部前端共用 ingest_uploads（增量/解析/OCR/限流全部一致）。"""
    if crud.get_kb_group(db, user_id=user.id, group_id=gid) is None:
        raise NotFoundError("分组不存在")
    # 入库链路会调 embedding（经 gateway 锁），同样纳入容量闸门
    return await _run_with_gate(
        lambda: ingest_uploads(db, user_id=user.id, group_id=gid, files=file)
    )


# ===== RAG 核心能力 =====


@router.post("/rag/retrieve")
async def retrieve_chunks(
    body: RetrieveRequest,
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """纯检索：返回切片原文 + 页码 + 相似度（不生成，调用方可自行组装 prompt）。"""
    group_ids = resolve_group_ids(db, user_id=user.id, group_ids=body.group_ids)
    chunks = await _run_with_gate(
        lambda: retrieve(
            body.query,
            user_id=user.id,
            group_ids=group_ids,
            top_k=body.top_k,
        )
    )
    return {
        "hits": [
            {
                "file_name": c.file_name,
                "page_no": c.page_no,
                "text": c.text,
                "score": round(float(c.score), 4),
                "group_id": c.group_id,
            }
            for c in chunks
        ],
        "count": len(chunks),
    }


@router.post("/rag/ask")
async def ask(
    body: AskRequestV1,
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """单轮 RAG 问答：检索 → 意图路由 → 生成，返回与内部 /chat/ask 相同的四字段契约。"""
    group_ids = resolve_group_ids(db, user_id=user.id, group_ids=body.group_ids)

    async def _do_ask():
        # 记忆召回/范围解析也要在闸门内：它们可能触发 embedding 调用
        ctx = await prepare_context(
            db, user_id=user.id, question=body.question, group_ids=group_ids
        )
        return await run_agent(
            question=body.question,
            user_id=user.id,
            group_ids=group_ids,
            history=[],  # 服务化 v1 无会话：多轮上下文由调用方维护
            guide_mode=body.guide_mode,
            page_range=ctx["page_range"],
            file_name=ctx["file_name"],
            scope_note=ctx["scope_note"],
            mem_context=ctx["mem_context"],
        )

    return await _run_with_gate(_do_ask)


@router.post("/rag/ask/stream")
async def ask_stream(
    body: AskRequestV1,
    user: User = Depends(get_current_app_user),
    db=Depends(get_db),
):
    """SSE 流式问答：{"t":"delta","v":…} / {"t":"done",…} / {"t":"error",…}。

    队列为什么放在闸门内：闸门的 acquire 必须由生成器自己完成——
    若在端点层先 acquire、生成器又被客户端断开而从未启动，容量会被永久占住；
    生成器内 acquire 起步 + finally 释放，没有泄漏窗口。
    因此排队超时只能以 error 事件（code=429）表达：流一旦开始，HTTP 头已发出。
    """
    group_ids = resolve_group_ids(db, user_id=user.id, group_ids=body.group_ids)

    async def _gen() -> AsyncIterator[str]:
        await _acquire_gate()

        async def _run() -> None:
            try:
                ctx = await prepare_context(
                    db, user_id=user.id, question=body.question, group_ids=group_ids
                )
                result = await run_agent(
                    question=body.question,
                    user_id=user.id,
                    group_ids=group_ids,
                    history=[],
                    guide_mode=body.guide_mode,
                    page_range=ctx["page_range"],
                    file_name=ctx["file_name"],
                    scope_note=ctx["scope_note"],
                    mem_context=ctx["mem_context"],
                    # 同步回调：put_nowait 非阻塞，不卡生成循环
                    on_delta=lambda s: queue.put_nowait(("delta", s)),
                )
            except Exception as e:  # noqa: BLE001 —— 流内必须以 error 事件收尾
                logger.exception("v1 SSE 问答流执行失败")
                message = e.message if hasattr(e, "message") else "服务器内部错误，请稍后重试"
                queue.put_nowait(("error", message))
            else:
                queue.put_nowait(("done", result))

        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_run())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "delta":
                    yield _sse({"t": "delta", "v": payload})
                elif kind == "done":
                    yield _sse({"t": "done", **payload})
                    return
                else:
                    yield _sse({"t": "error", "message": payload})
                    return
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            _v1_gate.release()

    return StreamingResponse(_gen(), media_type="text/event-stream")
