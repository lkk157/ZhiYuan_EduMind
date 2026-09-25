# -*- coding: utf-8 -*-
"""
记忆接口（M5）：生成/读取学习周报 + 记忆清单。

生成是 M5 唯一新增的 LLM 调用点（按钮触发）——照旧走 gateway 串行闸门；
解析失败翻译 502 人话（宁可报错不给假报告，与判分同一纪律）。
读接口全部带 user_id 过滤，非本人数据一律不可达。
"""
from fastapi import APIRouter, Depends

from app.api.auth import get_current_user
from app.core.exceptions import UpstreamError
from app.db import crud
from app.db.models import User
from app.db.session import get_db
from app.memory import long_term

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/report")
async def generate_report(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """生成本周学情报告（LLM 提炼对话+答题 → 报告双写入库），返回报告与薄弱点。"""
    try:
        return await long_term.generate_report(db, user_id=user.id)
    except RuntimeError as e:
        # 周报 JSON 解析失败 → 统一出口 502（人话可重试，绝不返回半截假报告）
        raise UpstreamError("学情报告生成失败，请重试") from e


@router.get("/report")
def latest_report(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """最近一次周报；从未生成过返回 report=None（前端据此显示引导态）。"""
    fact = crud.latest_memory_fact(db, user_id=user.id, kind=long_term.KIND_REPORT)
    if fact is None:
        return {"report": None, "created_at": None}
    return {"report": fact.content, "created_at": fact.created_at}


@router.get("/facts")
def list_facts(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """本人记忆清单（含薄弱点/洞察/报告条目），时间倒序——个性化「看得见」。"""
    facts = crud.list_memory_facts(db, user_id=user.id, limit=50)
    return [
        {
            "id": f.id,
            "kind": f.kind,
            "content": f.content,
            "ref_file": f.ref_file,
            "created_at": f.created_at,
        }
        for f in facts
    ]
