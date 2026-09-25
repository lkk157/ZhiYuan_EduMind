# -*- coding: utf-8 -*-
"""
监控看板接口（M6）：GET /monitoring/overview —— 统计数据给看板页。

接口为什么这么薄：聚合逻辑全部在 monitoring/stats.py（可单测的纯函数层），
本文件只做「鉴权 + 注入会话 + 调能力 + 返回」——与 api 层薄、能力层厚的
分层纪律一致（改统计口径不碰路由，改路由不碰统计）。

为什么按 user_id 隔离（而不是全站管理员视角）：
全项目越权纪律是「一切数据读写先过 user_id」（分组/文档/会话/错题/记忆全部如此），
看板没有理由破例；单机演示下 demo 账号的活动即全部活动，答辩叙事两不误。
"""
from fastapi import APIRouter, Depends

from app.api.auth import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.monitoring import stats

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/overview")
def get_overview(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """本人数据看板总览：提问量/近7天趋势、命中率/兜底、高频课件、答题表现、知识库概览。

    契约（前端与单测共用）：
    {"questions": {"total","last7","trend":[{"date":"MM-DD","count"}, …7项]},
     "answers": {"total","hit","fallback","hit_rate"},
     "top_files": [{"file_name","count"}, …],
     "quiz": {"total","correct","wrong","accuracy"},
     "kb": {"groups","documents","chunks"}}
    """
    return stats.overview(db, user_id=user.id)
