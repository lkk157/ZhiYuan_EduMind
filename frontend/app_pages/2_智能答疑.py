# -*- coding: utf-8 -*-
"""
智能答疑页：选分组提问 → 带溯源作答（会话内对话历史）。

为什么历史只存会话内（session_state）：M2 是无状态问答（后端 /chat/ask 不落库），
跨会话持久化属于长效记忆阶段的交付物——先演示闭环，不提前扩需求（CLAUDE.md §2）。

为什么未命中样式是 info 不是 error：兜底不是出错，而是「宁可不说也不编」的产品承诺
（后端空检索分支根本不会调用 LLM）——UI 口径与后端卖点保持一致。
"""
import streamlit as st

from components.sources import render_hit_badge, render_sources
from services.api import ApiClient, ApiError

# 鉴权守卫（纵深防御第二层，理由同上传管理页）
if not st.session_state.get("token"):
    st.warning("请先登录后再使用智能答疑。")
    st.stop()

client = ApiClient(token=st.session_state.token)

try:
    groups = client.list_groups()
except ApiError as e:
    st.error(e.message)
    st.stop()

if not groups:
    st.info("知识库还没有内容。请先到「上传管理」新建分组并上传课件。")
    st.stop()

# ---------- 检索范围（分组多选）----------
st.subheader("检索范围")
name_to_id = {g["name"]: g["id"] for g in groups}
selected = st.multiselect(
    "选择要检索的分组",
    list(name_to_id.keys()),
    default=list(name_to_id.keys()),
)
if not selected:
    st.caption("请至少选择一个分组作为检索范围。")

# ---------- 对话区 ----------
st.subheader("对话")
history = st.session_state.chat_history
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_hit_badge(msg.get("hit", True))
            render_sources(msg.get("sources") or [])

if history:
    if st.button("清空对话", icon=":material/delete:"):
        st.session_state.chat_history = []
        st.rerun()

# submit_mode="disable"：回答生成期间禁用输入框——防止连发把 7B 推理队列打成长龙
# （后端虽有互斥闸门兜底，前端体验上也不该让用户误以为「卡了」）
prompt = st.chat_input("就课件内容提问（支持术语/对比问法）", submit_mode="disable")
if prompt:
    if not selected:
        st.warning("请先选择检索分组再提问。")
        st.stop()

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # group_ids 显式传选中分组（后端还会逐个校验归属，防越权检索）
            result = client.ask(prompt, group_ids=[name_to_id[n] for n in selected])
            st.markdown(result["answer"])
            render_hit_badge(result["hit"])
            render_sources(result.get("sources") or [])
            history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources") or [],
                    "hit": result["hit"],
                }
            )
        except ApiError as e:
            # 失败也进历史：会话脉络不断线，用户知道哪一轮出了问题
            st.error(e.message)
            history.append({"role": "assistant", "content": f"请求失败：{e.message}", "sources": [], "hit": False})
