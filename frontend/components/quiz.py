# -*- coding: utf-8 -*-
"""
试题卡组件（M4）：把出题工具返回的 JSON 渲染成交互卡片（作答 → 判分）。

为什么做成组件而不是写死在答疑页：出题结果既出现在「刚生成的这条回答」里，
也会出现在「切换会话回看的历史消息」里——两处渲染逻辑必须同一份，
否则回看时题目显示成一坨 JSON 原文（答辩现场最尴尬的穿帮）。

为什么作答态存 session_state 而不是组件内部变量：Streamlit 每次交互都整页重跑，
局部变量活不过一轮 rerun——按 key 存进 session_state，选项点了才不会丢。
"""
import json

import streamlit as st

from services.api import ApiError


def try_parse_quiz(content: str) -> dict | None:
    """消息内容 → 试题 JSON；不是合法试题返回 None（调用方按普通文本渲染）。

    为什么用「内容特征」识别试题而不用数据库存 intent：message 表没有 intent 列，
    给已存在的表加列需要 SQLite/MySQL 双份迁移（create_all 不改已有表）——
    JSON 前缀识别零迁移成本，代价是「识别依赖格式稳定」，由 parse_quiz 的
    严格结构校验兜底（半残 JSON 不会误入卡片分支）。
    """
    text = (content or "").strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("type") == "quiz" and isinstance(obj.get("questions"), list):
        return obj
    return None


def render_quiz_card(
    quiz: dict,
    *,
    key_prefix: str,
    client,
) -> None:
    """渲染一张可作答的试题卡（key_prefix 隔离多张卡的 widget 状态）。

    判分结果存 session_state[key_prefix_result]：rerun 后仍展示，
    直到下一次点「重新判分」覆盖。
    """
    questions = quiz.get("questions") or []
    st.markdown(":material/quiz: **随堂测**")
    st.caption(f"共 {len(questions)} 题 · 作答后点「提交判分」")

    # 收集作答：choice 用 radio、short 用文本框，key 按题号隔离
    answers: list[str] = []
    for i, q in enumerate(questions):
        with st.container(border=True):
            qtype = q.get("type")
            stem = q.get("question", "")
            if qtype == "choice":
                options = [str(o) for o in q.get("options") or []]
                std = str(q.get("answer", "")).upper()
                if len(std) > 1:
                    # 多选题（answer 为多个字母，如 "AB"）→ checkbox 组，可多选——
                    # 2026-09-25 反馈的修复点：radio 单选框容不下多选题
                    picked = st.multiselect(
                        f"第 {i + 1} 题（多选）. {stem}",
                        options,
                        key=f"{key_prefix}_q{i}",
                    )
                    letters = sorted(
                        chr(ord("A") + options.index(p)) for p in picked if p in options
                    )
                    answers.append("".join(letters))
                else:
                    # index=None：未选择时返回 None（Streamlit 1.64 原生支持，无 accept_none 参数）——
                    # 未作答按空串送判分（单选后端计 0 分）
                    picked = st.radio(
                        f"第 {i + 1} 题. {stem}",
                        options,
                        index=None,
                        key=f"{key_prefix}_q{i}",
                    )
                    # 取选项字母（A/B/C/D）而不是整句文本：与标准答案格式对齐
                    answers.append(
                        chr(ord("A") + options.index(picked)) if picked in options else ""
                    )
            else:  # short 简答
                ans = st.text_input(
                    f"第 {i + 1} 题. {stem}",
                    key=f"{key_prefix}_q{i}",
                )
                answers.append((ans or "").strip())

    col_btn, col_result = st.columns([1, 2])
    with col_btn:
        if st.button("提交判分", key=f"{key_prefix}_submit", icon=":material/check:"):
            try:
                result = client.score_quiz(quiz, answers)
                st.session_state[f"{key_prefix}_result"] = result
            except ApiError as e:
                st.error(e.message)
    with col_result:
        result = st.session_state.get(f"{key_prefix}_result")
        if result:
            score = result.get("score")
            comment = result.get("comment", "")
            st.markdown(f"**:material/grade: {score} 分** — {comment}")

    # 标准答案折叠展示：作答判分后再看，避免「对着答案做题」
    with st.expander("查看标准答案与解析"):
        for i, q in enumerate(questions):
            st.markdown(f"**第 {i + 1} 题**：{q.get('answer', '')}")
            if q.get("explanation"):
                st.caption(f"解析：{q['explanation']}")
