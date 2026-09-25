# -*- coding: utf-8 -*-
"""
数据看板页（M6）：提问热度/命中率/高频课件/答题表现/知识库概览。

为什么看板单独一页而不是塞进「错题与学情」：受众不同——错题页是学生视角的
个人学习闭环，看板是「系统运行得怎么样」的监控视角（答辩时演示给评委看的
运行数据）；混在一页里两拨指标互相稀释。

数据全部来自 GET /monitoring/overview（后端从既有表现算）：页面零计算、
零模型调用，打开即出图。
"""
import pandas as pd
import streamlit as st

from services.api import ApiClient, ApiError

# 鉴权守卫（纵深防御第二层，与其余页面同款）
if not st.session_state.get("token"):
    st.warning("请先登录后再使用数据看板。")
    st.stop()

client = ApiClient(token=st.session_state.token)

try:
    ov = client.monitoring_overview()
except ApiError as e:
    st.error(e.message)
    st.stop()

questions = ov.get("questions", {})
answers = ov.get("answers", {})
quiz = ov.get("quiz", {})
kb = ov.get("kb", {})

st.subheader("数据看板")
st.caption(
    "统计口径：命中=回答有课件依据并附【来源】；兜底=检索未命中、系统按防幻觉闸门拒答不编造。"
    "数据来自问答/答题记录与知识库元数据的实时聚合。"
)

# ---------- 提问与命中 ----------
st.markdown("### 问答概况")
c1, c2, c3, c4 = st.columns(4)
c1.metric("累计提问", questions.get("total", 0))
c2.metric("近 7 天提问", questions.get("last7", 0))
c3.metric("命中率", f"{float(answers.get('hit_rate', 0)) * 100:.1f}%")
c4.metric("兜底次数", answers.get("fallback", 0))

# 近 7 天趋势：pandas 是 Streamlit 的自带依赖（零新增 requirements 项），
# set_index 后直接喂 st.bar_chart——x 轴连续 7 天（后端已补 0 值桶）
trend = questions.get("trend") or []
if trend:
    df = pd.DataFrame(trend)
    st.bar_chart(df.set_index("date")["count"], height=220)

# ---------- 高频知识点 ----------
st.markdown("### 高频知识点（被引用最多的课件）")
top_files = ov.get("top_files") or []
if top_files:
    # 文件名长、条数少（Top 5）：列表比柱状图更可读，也不会被截断
    for item in top_files:
        st.markdown(f"- **{item['file_name']}** — 被 {item['count']} 次回答引用")
else:
    st.caption("还没有命中过知识库的问答——提问命中后这里会按课件热度排序。")

# ---------- 答题表现与知识库概览 ----------
st.markdown("### 学习与知识库")
row1 = st.columns(4)
row1[0].metric("累计答题", quiz.get("total", 0))
row1[1].metric("答题正确率", f"{float(quiz.get('accuracy', 0)) * 100:.1f}%")
row1[2].metric("错题数", quiz.get("wrong", 0))
row1[3].metric("课件文档", kb.get("documents", 0))

row2 = st.columns(3)
row2[0].metric("知识分组", kb.get("groups", 0))
row2[1].metric("总切块数", kb.get("chunks", 0))
row2[2].metric("有效回答", answers.get("total", 0))
