# -*- coding: utf-8 -*-
"""
问答接口：/chat/ask（M4 起为 Agent 状态图的 HTTP 出口）+ /chat/conversations* + /chat/score。

分层（M4 重构后）：本文件只做「鉴权/会话/分组校验 → 调 run_agent → 落库 → 拼响应」——
意图分类、query 改写、检索、四个工具全部收在 app/agent/（状态图），
防幻觉硬闸门/三层防线的现行宿主是 agent/graph.py 与 agent/tools.py，
本文件不再直接碰 gateway/retrieve（测试补丁缝随之下迁，见 tests 的 patch 命名空间）。

会话历史（2026-09-24）：/chat/conversations* 四个端点 + ask 可选 conversation_id 落库；
落库内容 = run_agent 的四字段结果（answer/sources/hit/intent 中 intent 不落库——
message 表无该列，加列需迁移，回看时按内容特征识别，见 M4 过程报告）。

显存红线自查：本文件零直接模型调用；run_agent 内部（改写/分类/生成）全部
经 gateway 的 Semaphore(1)+Lock 串行排队，单轮最多 3 次顺序调用，无并发推理。
"""
import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import run_agent
from app.agent.scope import resolve_scope
from app.agent.tools import score_quiz
from app.api.auth import get_current_user
from app.core.exceptions import AppError, NotFoundError, UpstreamError
from app.memory.long_term import KIND_WEAK, make_weak_fact, recall_memories, write_fact
from app.memory.short_term import load_window
from app.db import crud
from app.db.models import User
from app.db.session import get_db

# 路由前缀 /chat：问答接口挂在它下面（层间契约的路由路径，禁止改动）
router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """提问入参。

    group_ids 可缺省：缺省=检索本人全部分组（最常见的「随便问」场景）；
    显式传了则只在指定分组里检索（比如「只问高数分组」），且逐个校验归属。

    conversation_id 可缺省（2026-09-24 会话历史改版引入）：
    缺省 = 不落库（无状态问答）——现有单测与 demo_e2e.py 零改动、行为完全不变；
    显式给了 = 一问一答写入该会话，且先校验归属（非本人 404 防探测）。
    「不传不保存」的语义边界比「不传就自动建会话」更安全：
    后者会让脚本类调用（demo/压测）每次请求都污染会话列表。
    """

    question: str
    group_ids: list[int] | None = None
    conversation_id: int | None = None
    # M4 引导式答疑（苏格拉底模式）：True 时 qa 工具改用引导式 system（默认关闭）
    guide_mode: bool = False
    # 体验增强包：True 时响应为 SSE 流（delta/done/error 事件），False=既有 JSON——
    # 默认 false 让老调用方（demo_e2e/既有单测）零感知
    stream: bool = False


class ConversationCreateRequest(BaseModel):
    """新建会话入参。title 允许超长（接口层截断到 128），
    避免 pydantic 422 把「首问很长」这种正常输入拒之门外。"""

    title: str = ""


def _require_conversation(db, *, user_id: int, conversation_id: int):
    """取本人会话，取不到（不存在或非本人）一律 NotFoundError（理由同 kb._require_group：
    报 403 等于承认 id 存在，可被枚举探测）。"""
    conversation = crud.get_conversation(db, user_id=user_id, conversation_id=conversation_id)
    if conversation is None:
        raise NotFoundError("会话不存在")
    return conversation


def _serialize_messages(messages) -> list[dict]:
    """Message 行 → 接口契约：sources JSON 字符串还原成 list[dict]，hit 保持 True/False/None。"""
    return [
        {
            "role": m.role,
            "content": m.content,
            "hit": m.hit,
            # 库里存 JSON 字符串（MySQL/SQLite 同构取舍），出口还原成结构化列表；
            # 解析失败按空列表容错（坏数据不许 500，历史读取永远可用）
            "sources": _load_sources(m.sources),
            "created_at": m.created_at,
        }
        for m in messages
    ]


def _load_sources(raw: str | None) -> list[dict]:
    """sources 列（JSON 字符串）→ list[dict]，坏数据静默降级为空列表。"""
    try:
        parsed = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


# ===== 会话历史 CRUD（挂 /chat 前缀：与问答同域，不另开路由树）=====


@router.post("/conversations", status_code=201)
def create_conversation(
    body: ConversationCreateRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """新建空会话，返回 {id, title}。

    标题兜底逻辑：空/纯空白 → 「新对话」；超 128 截断（前端通常已截 30 字，
    这里是服务端复校——客户端限制不是安全边界，与批量上传同款纪律）。
    """
    title = (body.title or "").strip()[:128] or "新对话"
    conversation = crud.create_conversation(db, user_id=user.id, title=title)
    return {"id": conversation.id, "title": conversation.title}


@router.get("/conversations")
def list_conversations(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """列本人全部会话（含消息数），按最近活跃倒序——前端侧边栏会话列表的数据源。"""
    rows = crud.list_conversations(db, user_id=user.id)
    return [
        {
            "id": c.id,
            "title": c.title,
            "message_count": count,
            "created_at": c.created_at,
            "updated_at": c.updated_at,
        }
        for c, count in rows
    ]


@router.get("/conversations/{cid}/messages")
def list_messages(
    cid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """取某会话全部消息（时间正序）。非本人/不存在 → 404（防探测）。"""
    _require_conversation(db, user_id=user.id, conversation_id=cid)
    return _serialize_messages(crud.list_messages(db, user_id=user.id, conversation_id=cid))


@router.delete("/conversations/{cid}")
def delete_conversation(
    cid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """删除会话（级联删消息，crud 手工级联保证 SQLite/MySQL 都不留孤儿）。"""
    _require_conversation(db, user_id=user.id, conversation_id=cid)
    crud.delete_conversation(db, user_id=user.id, conversation_id=cid)
    return {"ok": True}


@router.post("/ask")
async def ask(
    body: AskRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """知识库问答（M4 起经 Agent 状态图）：校验 → run_agent → 可选落库 → 返回。

    返回 {answer, sources, hit, intent, conversation_id}（intent ∈ qa/quiz/summary/calc）。
    hit=false 的口径不变：answer 恒为 FALLBACK_MESSAGE、sources 恒为空——
    三条触发路径（空检索零调用 / 资料不足闸门 / 出题 JSON 解析失败）统一在
    agent 层的 _fallback 收口，本接口只负责原样透传 + 落库。
    """
    # 0) 会话归属先校验（fail fast）：拿别人的 conversation_id 提前 404，
    #    别等检索/生成烧完一轮显存才发现没权限——浪费推理还把越权拖到耗时操作之后
    conversation = None
    if body.conversation_id is not None:
        conversation = _require_conversation(
            db, user_id=user.id, conversation_id=body.conversation_id
        )

    # 1) 分组权限边界：缺省=本人全部分组；显式给了必须逐个确认归属，
    #    否则借别人 group_id 提问等于跨用户检索（数据泄漏），一律 NotFoundError 防探测
    if body.group_ids is None:
        group_ids = [g.id for g in crud.list_kb_groups(db, user_id=user.id)]
    else:
        group_ids = []
        for gid in body.group_ids:
            if crud.get_kb_group(db, user_id=user.id, group_id=gid) is None:
                # 非本人/不存在统一 404：不泄漏「这个分组存在但不归你」
                raise NotFoundError("分组不存在")
            if gid not in group_ids:  # 去重：重复的 gid 会把同批命中翻倍进入排序
                group_ids.append(gid)

    # 2) 滑窗读历史（仅会话问答有历史；无历史时后续改写零模型调用）
    history = load_window(db, user_id=user.id, conversation_id=body.conversation_id)

    # 2.3) 长效记忆召回（M5 个性化）：无记忆快路零开销；语义段失败自动降级最近条——
    #      注入是增强，绝不允许它反过来让提问失败（见 long_term.recall_memories）
    mem_context = await recall_memories(db, user_id=user.id, question=body.question)

    # 2.5) 范围解析（「第X-Y页」「第N章」→ 检索过滤条件；无范围词时零开销返回 None）。
    #     放接口层是因为它要读 DB（文档文件路径）——图内拿不到会话外的事实源。
    scope, scope_note = resolve_scope(
        db, user_id=user.id, group_ids=group_ids, question=body.question
    )
    page_range = scope.get("page_range") if scope else None
    scope_file = scope.get("file_name") if scope else None

    # 3) 分流：SSE 流式 / 既有 JSON（校验都在上面完成——404 类错误在流开始前就返回）
    if body.stream:
        return StreamingResponse(
            _ask_sse(
                db,
                conversation=conversation,
                body=body,
                user_id=user.id,
                group_ids=group_ids,
                history=history,
                page_range=page_range,
                scope_file=scope_file,
                scope_note=scope_note,
                mem_context=mem_context,
            ),
            media_type="text/event-stream",
        )

    # 3N) Agent 状态图：计算旁路/改写 → 检索（带范围过滤） → 分类 → 四工具（布局见 agent/graph.py）
    result = await run_agent(
        question=body.question,
        user_id=user.id,
        group_ids=group_ids,
        history=history,
        guide_mode=body.guide_mode,
        page_range=page_range,
        file_name=scope_file,
        scope_note=scope_note,
        mem_context=mem_context,
    )
    _persist_result(db, conversation, body, result)
    return {**result, "conversation_id": body.conversation_id}


def _persist_result(db, conversation, body, result: dict) -> None:
    """会话落库（JSON/SSE 两分支共用，杜绝一条落库一条忘的漂移）：
    hit=false 的兜底轮同样落库（历史不许选择性失忆）；
    sources 用 ensure_ascii=False 存原文——运维 SELECT * 直接可读。"""
    if conversation is not None:
        crud.append_message_pair(
            db,
            conversation=conversation,
            question=body.question,
            answer=result["answer"],
            hit=result["hit"],
            sources_json=json.dumps(result["sources"], ensure_ascii=False),
        )


def _sse(obj: dict) -> str:
    """dict → SSE data 行（ensure_ascii=False：中文 token 原样上屏，转义会伤打字机观感）。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _ask_sse(
    db,
    *,
    conversation,
    body: AskRequest,
    user_id: int,
    group_ids: list[int],
    history: list[dict],
    page_range,
    scope_file,
    scope_note,
    mem_context: list[str] | None = None,
) -> AsyncIterator[str]:
    """SSE 问答流：后台跑 Agent，token 经队列回吐成 delta 事件，收尾发 done/error。

    为什么用「后台任务 + 队列」而不是边跑边 yield：run_agent 是一次 await 调用，
    中途拿不到 token——把它的 on_delta 回调接进队列，生成器这端就能边收边发。
    这不是并发推理：on_delta 只是把 token 塞进内存队列，唯一的 generate 仍在
    gateway 锁内从头流到尾（红线自查通过）。

    事件契约（前端按此分发）：
      {"t":"delta","v":"…"}   逐字增量（仅 qa/总结有；出题/计算直接等 done）
      {"t":"done", …run_agent 四字段, "conversation_id":…}  终局结果=净化后契约
      {"t":"error","message":"…"}  出错（不再有 done）
    落库发生在 done 之前——前端收到 done 时服务端已写完，回看无时差。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def _run() -> None:
        try:
            result = await run_agent(
                question=body.question,
                user_id=user_id,
                group_ids=group_ids,
                history=history,
                guide_mode=body.guide_mode,
                page_range=page_range,
                file_name=scope_file,
                scope_note=scope_note,
                mem_context=mem_context,
                # 同步回调直塞队列：put_nowait 非阻塞，不卡生成循环
                on_delta=lambda s: queue.put_nowait(("delta", s)),
            )
        except Exception as e:  # noqa: BLE001 —— 流内必须以 error 事件收尾，不能让 SSE 裸断
            logger.exception("SSE 问答流执行失败")
            message = e.message if hasattr(e, "message") else "服务器内部错误，请稍后重试"
            queue.put_nowait(("error", message))
        else:
            queue.put_nowait(("done", result))

    task = asyncio.create_task(_run())
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "delta":
                yield _sse({"t": "delta", "v": payload})
            elif kind == "done":
                _persist_result(db, conversation, body, payload)
                yield _sse(
                    {"t": "done", **payload, "conversation_id": body.conversation_id}
                )
                return
            else:  # error
                yield _sse({"t": "error", "message": payload})
                return
    finally:
        # 消费端提前 return（前端断连等）：任务若还在跑就取消——
        # 不取消会继续白跑一次完整生成还往没人听的队列里塞 token
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


class ScoreRequest(BaseModel):
    """判分入参：quiz=出题工具返回的试题 JSON 对象，answers=按题序的学生作答文本。

    sources（M5）：本次试题的来源卡片（前端随判分一并提交）——错题的教材锚点，
    知识点图谱「错题→章节→关联推荐」链路的起点；缺省无锚则推荐退化为全局结构边。
    """

    quiz: dict
    answers: list[str]
    sources: list[dict] = []


@router.post("/score")
async def score(
    body: ScoreRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """试题判分：单选代码层精确判、简答 LLM 按要点判（见 tools.score_quiz）。

    M5 起判分即落库：**全部作答**写 quiz_records（错题本按 is_correct 过滤，
    周报要含答对的算正确率）；**答错的**另生成一条零 LLM 的薄弱记忆（双写）。
    教材锚取 sources[0]（本次试题的首要来源）；返回 recorded=错题数（前端提示用）。
    单选全卷零 LLM 调用（能确定的绝不交给模型）；简答解析失败报 502 人话可重试，
    宁可报错也不返回假分数——解析失败时不落库（没判明白就不记录，防脏数据）。
    """
    questions = body.quiz.get("questions") if isinstance(body.quiz, dict) else None
    if not isinstance(questions, list) or not questions:
        raise AppError("试题格式不正确", code=400)
    try:
        result = await score_quiz(questions, body.answers)
    except RuntimeError as e:
        # LLM 判分输出解析失败：翻译成统一出口的人话 502（前端 ApiError 直接透出）
        raise UpstreamError("判分失败，请重试") from e

    # 教材锚（图谱链路起点）：sources[0] 优先，无来源则无锚（推荐退化为全局结构边）
    anchor = body.sources[0] if body.sources else {}
    ref_file = str(anchor.get("file_name", "") or "")
    ref_page = int(anchor.get("page_no", 0) or 0)
    recorded = 0
    for i, detail in enumerate(result.get("details", [])):
        q = questions[i]
        answers_i = body.answers[i] if i < len(body.answers) else ""
        crud.create_quiz_record(
            db,
            user_id=user.id,
            question=str(q.get("question", "")),
            question_type=str(q.get("type", "choice")),
            user_answer=str(answers_i),
            correct_answer=str(q.get("answer", "")),
            explanation=str(q.get("explanation", "")),
            is_correct=bool(detail["correct"]),
            ref_file=ref_file,
            ref_page=ref_page,
        )
        if not detail["correct"]:
            recorded += 1
            # 薄弱记忆零 LLM 模板生成 + 双写（MySQL 先行，索引 best-effort）
            await write_fact(
                db,
                user_id=user.id,
                kind=KIND_WEAK,
                content=make_weak_fact(str(q.get("question", "")), ref_file),
                ref_file=ref_file,
            )
    return {"score": result["score"], "comment": result["comment"], "recorded": recorded}
