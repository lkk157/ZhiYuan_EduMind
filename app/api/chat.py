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
"""
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
    """

    question: str
    group_ids: list[int] | None = None


@router.post("/ask")
async def ask(
    body: AskRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """知识库问答：检索 → （命中）生成带溯源的答案 / （未命中）兜底且不调 LLM。

    返回 {answer, sources:[{file_name,page_no,snippet}], hit}。
    hit=false 时 answer 恒为 FALLBACK_MESSAGE、sources 恒为空——防幻觉硬闸门所在。
    """
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
    if not chunks:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "hit": False}

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
    return {"answer": answer, "sources": sources, "hit": True}
