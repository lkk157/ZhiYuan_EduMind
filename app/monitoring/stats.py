# -*- coding: utf-8 -*-
"""
统计聚合（M6）：监控看板的数据层——只读既有表，零写入、零模型调用。

为什么不建 ROADMAP 早期预告的 qa_logs / retrieval_logs 日志表（2026-09-25 产品确认）：
1. messages 表本就落了每次提问的内容/时间/是否命中/来源 JSON——它本身就是
   结构化的问答日志，再插一份 qa_logs 是双写同一件事，答辩会被追问「两张表记一样的东西」；
2. 检索日志同理：回答的 sources 就是「检索了什么、命中了哪一页」的落库形态；
3. 新表从零开始，看板大半是空的；既有数据（M2 以来全部问答）现算即刻可见；
4. 不碰问答热路径 = 不给 ask 链路加任何写库动作，显存红线自然不沾边。
（ROADMAP 目录树与计划变更记录已同步改口径。）

为什么消息明细在 Python 端聚合而不是 SQL GROUP BY：
单机演示体量小（千行级），拉行现算只需写一处；SQL 日期分组要处理
MySQL/SQLite 方言差异（DATE() 函数、server_default 时区语义），跨库测试难写。
体量上万再改 SQL——README 对比表的升级路径叙事覆盖此决策。
"""
import json
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Conversation, Document, KbGroup, Message, QuizRecord

# 趋势窗口：近 7 天（含今天）——答辩口径「最近一周的提问热度」
TREND_DAYS = 7
# 高频知识点 Top N：看板只展示最热的几个，全量列表对演示没有信息增量
TOP_FILES_N = 5


def _parse_sources(raw: str | None) -> list[dict]:
    """解析回答行的 sources JSON；脏数据返回空列表而不是抛异常。

    为什么容错：sources 是 Text 列自由写入，历史脏行（手工改库/早期契约）
    不该让整个看板 500——看板是答辩门面，宁可少统计一条也不能崩。
    """
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return data if isinstance(data, list) else []


def overview(db: Session, *, user_id: int, now: datetime | None = None) -> dict:
    """本人全量统计总入口（看板唯一数据源）。

    now 可注入：测试固定时钟，断言趋势分桶不受「跑测试的瞬间跨没跨午夜」影响；
    生产缺省本机当前时间（单机 MySQL 同区，与 created_at 的 now() 同源）。

    口径（答辩照读）：
    - questions.total / answers.* / top_files / quiz / kb = 全量（演示体量小，全量最直观）；
    - questions.last7 / trend = 近 7 天窗口（看「最近热度」用）；
    - hit_rate = 命中回答 / 全部有效回答；兜底（hit=false）计入分母不计入分子。
    """
    now = now or datetime.now()

    # ---- 本人全部消息：JOIN 会话表取归属（messages 自身没有 user_id 列）----
    # 单列 select + join：Message 只在外键上与 Conversation 关联，显式 ON 条件最直白
    messages = list(
        db.execute(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .order_by(Message.id)
        )
        .scalars()
    )

    # ---- 提问量与近 7 天趋势（按天分桶）----
    day_counts: Counter = Counter()
    questions_total = 0
    for m in messages:
        if m.role != "user":
            continue
        questions_total += 1
        if m.created_at is None:
            continue
        day_counts[m.created_at.date()] += 1

    # 补齐窗口内每一天（没有提问的日子补 0）：折线/柱状图 x 轴连续，缺桶会把日期错位
    window_start = (now - timedelta(days=TREND_DAYS - 1)).date()
    trend: list[dict] = []
    for i in range(TREND_DAYS):
        d = window_start + timedelta(days=i)
        trend.append({"date": d.strftime("%m-%d"), "count": day_counts.get(d, 0)})
    last7 = sum(day_counts[d] for d in day_counts if d >= window_start)

    # ---- 命中率 / 兜底（只统计助手回答：hit 非 NULL 才有命中语义）----
    answers_total = hit_count = 0
    file_counter: Counter = Counter()
    for m in messages:
        if m.role != "assistant" or m.hit is None:
            continue
        answers_total += 1
        if m.hit:
            hit_count += 1
        # 高频知识点：命中回答引用了哪些课件（fallback 行 sources=[] 天然不计）
        for item in _parse_sources(m.sources):
            name = str(item.get("file_name") or "").strip()
            if name:
                file_counter[name] += 1

    # count 降序、同数按文件名升序：结果确定可断言（Counter.most_common 对并列不稳定）
    top_files = [
        {"file_name": name, "count": count}
        for name, count in sorted(file_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_FILES_N]
    ]

    # ---- 答题表现（quiz_records：判分时已逐题落库）----
    quiz_total = db.execute(
        select(func.count(QuizRecord.id)).where(QuizRecord.user_id == user_id)
    ).scalar() or 0
    quiz_correct = db.execute(
        select(func.count(QuizRecord.id)).where(
            QuizRecord.user_id == user_id, QuizRecord.is_correct.is_(True)
        )
    ).scalar() or 0

    # ---- 知识库概览（分组数 / 文档数 / 总切块数）----
    kb_groups = db.execute(
        select(func.count(KbGroup.id)).where(KbGroup.user_id == user_id)
    ).scalar() or 0
    kb_docs, kb_chunks = db.execute(
        select(func.count(Document.id), func.coalesce(func.sum(Document.chunk_count), 0)).where(
            Document.user_id == user_id
        )
    ).one()

    # 比率统一 round 到 4 位、零数据返回 0.0（除零保护——新用户看板不能 500）
    hit_rate = round(hit_count / answers_total, 4) if answers_total else 0.0
    accuracy = round(quiz_correct / quiz_total, 4) if quiz_total else 0.0

    return {
        "questions": {
            "total": questions_total,
            "last7": last7,
            "trend": trend,
        },
        "answers": {
            "total": answers_total,
            "hit": hit_count,
            "fallback": answers_total - hit_count,
            "hit_rate": hit_rate,
        },
        "top_files": top_files,
        "quiz": {
            "total": quiz_total,
            "correct": quiz_correct,
            "wrong": quiz_total - quiz_correct,
            "accuracy": accuracy,
        },
        "kb": {
            "groups": kb_groups,
            "documents": int(kb_docs or 0),
            "chunks": int(kb_chunks or 0),
        },
    }
