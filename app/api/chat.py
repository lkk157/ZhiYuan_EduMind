# -*- coding: utf-8 -*-
"""
问答接口：/chat/ask（M2 读路径的 HTTP 出口）。

为什么「命中为空就绝不调 LLM」写在接口层（本文件的命门）：
召回为空时任何生成都是无中生有——通用 RAG 产品在这里会让模型「礼貌性地编一段」，
本项目的核心卖点恰恰是反着来：宁可回 FALLBACK_MESSAGE 兜底话术，也绝不把
无关/缺失的资料喂给 LLM 去编。防幻觉不是提示词里的恳求，而是代码路径上的硬闸门：
空命中分支根本不会走到 gateway.generate。答辩演示就讲这条分支。

其余出口纪律：
- 模型自写的【来源…】一律 sanitize_answer 删掉（那是幻觉重灾区）；
- 真来源由 append_sources 按真实命中统一追加，sources 列表与它同序同去重，
  保证前端展示的出处与答案末尾的【来源】永远一致。

会话历史（2026-09-24）：/chat/conversations* 四个端点 + ask 可选 conversation_id 落库。
落库只是「记录」不改变生成路径——检索仍旧只看当前 question（多轮上下文/query 改写属 M4），
本文件新增的全部是 DB 读写，零新增模型调用（显存红线自查通过）。
"""
import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.core.llm import gateway
from app.db import crud
from app.db.models import User
from app.db.session import get_db
from app.rag.prompts import (
    FALLBACK_MESSAGE,
    append_sources,
    build_qa_prompt,
    sanitize_answer,
)
from app.rag.retriever import retrieve

# 路由前缀 /chat：问答接口挂在它下面（层间契约的路由路径，禁止改动）
router = APIRouter(prefix="/chat", tags=["chat"])

# 溯源摘要长度：snippet=块文本前约 80 字（契约口径），够用户核对出处又不撑爆响应体
SNIPPET_LEN = 80


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
    """知识库问答：检索 → （命中）生成带溯源的答案 / （未命中）兜底且不调 LLM。

    返回 {answer, sources:[{file_name,page_no,snippet}], hit, conversation_id}。
    hit=false 时 answer 恒为 FALLBACK_MESSAGE、sources 恒为空——防幻觉硬闸门所在。
    conversation_id 原样回显（None=本次未落库）。
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

    # 2) 检索：阈值过滤 + top_k 截断在 retriever 内完成（宁缺毋滥是防幻觉第一道闸）
    chunks = await retrieve(body.question, user_id=user.id, group_ids=group_ids)

    # 3) ★ 防幻觉硬闸门：未命中立即兜底返回，严禁调用 gateway.generate——
    #    没有资料还让 7B 作答 = 百分之百编造；兜底话术引导用户换问法或先传资料。
    #    （本分支是否真的不碰 LLM 是单测的重点断言，改动前先想清楚卖点还在不在）
    #    兜底问答同样落库：用户回看时必须看到「这轮没命中」，历史不许选择性失忆。
    if not chunks:
        if conversation is not None:
            crud.append_message_pair(
                db,
                conversation=conversation,
                question=body.question,
                answer=FALLBACK_MESSAGE,
                hit=False,
                sources_json="[]",
            )
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "hit": False,
            "conversation_id": body.conversation_id,
        }

    # 4) 命中：编排提示词 → 生成 → 净化伪来源 → 强制追加真来源
    system, prompt = build_qa_prompt(body.question, chunks)
    raw = await gateway.generate(
        model=settings.llm_model,  # 模型名只从配置读，禁止硬编码（CLAUDE.md §5）
        prompt=prompt,
        system=system,
        num_predict=512,  # 输出长度受控（显存红线），答案不做长文生成
        temperature=0.3,  # 低温度：问答要忠于资料，减少发挥
    )
    answer = append_sources(sanitize_answer(raw), chunks)

    # 5) sources 与 append_sources 同序同去重（(file_name, page_no) 去重保序），
    #    保证前端出处列表 == 答案末尾【来源】；snippet 取该块前约 80 字供用户核对
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.file_name, chunk.page_no)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file_name": chunk.file_name,
                "page_no": chunk.page_no,
                "snippet": chunk.text[:SNIPPET_LEN],
            }
        )

    # 6) 会话落库（可选）：sources 用 ensure_ascii=False 存原文——
    #    中文不转 \uXXXX，运维直接 SELECT * 也能读懂，回看还原零损耗
    if conversation is not None:
        crud.append_message_pair(
            db,
            conversation=conversation,
            question=body.question,
            answer=answer,
            hit=True,
            sources_json=json.dumps(sources, ensure_ascii=False),
        )
    return {
        "answer": answer,
        "sources": sources,
        "hit": True,
        "conversation_id": body.conversation_id,
    }
