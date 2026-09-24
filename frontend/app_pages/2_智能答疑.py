# -*- coding: utf-8 -*-
"""
智能答疑页：选分组提问 → 带溯源作答（会话历史持久化，可切换回看）。

会话历史设计（2026-09-24 起）：
- MySQL 是事实源：每个会话的问答经 /chat/ask?conversation_id= 落库，
  刷新页面/换设备重新登录都能回看（旧版只存 session_state，一刷就丢）；
- 本页 session_state.chat_history 只是「当前会话的渲染缓存」——
  切换会话时从服务端整拉覆盖，新建会话时清空，任何时候以服务端为准；
- 「＋ 新对话」选择哨兵：首问时才真正建会话行（标题=首问截 30 字，零 LLM 调用），
  避免点一下「新建」就留一个永远空着的会话壳。

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

# ---------- 侧边栏：会话管理 ----------
# 放在主区之前执行：切换会话要先拉服务端消息覆盖 chat_history，主区才渲染得对
with st.sidebar:
    st.subheader("对话会话")
    try:
        conversations = client.list_conversations()
    except ApiError as e:
        st.error(e.message)
        st.stop()

    # id → 会话行：format_func 只拿得到值（id/None），展示字段要从这里查
    by_id = {c["id"]: c for c in conversations}

    def _fmt_conv(value) -> str:
        """选择器显示文案：None=新建哨兵；正常会话显示 标题（N 条）。"""
        if value is None:
            return "＋ 新对话"
        c = by_id.get(value)
        return f"{c['title']}（{c['message_count']} 条）" if c else str(value)

    # 选项用 id（int/None）而不是会话 dict：session_state 里存的是值本身，
    # dict 每次拉取都是新对象、message_count 还会变，存 dict 会让选择态因「值不相等」而失焦
    options = [None] + [c["id"] for c in conversations]

    # —— 回填同步（必须在 selectbox 实例化之前）——
    # 新建会话/删除会话发生在主区（widget 已实例化，之后改 st.session_state[key] 会抛
    # StreamlitAPIException），只能留到下一次重跑、在实例化前把选择器拉齐到当前会话：
    #   chat_conv_id=X 而 selector 还是 None → 回填 X（新建后选中新会话）
    #   chat_conv_id=None 而 selector 还是旧 id → 回填 None（删除/新对话后回到哨兵）
    if st.session_state.get("chat_conv_id") != st.session_state.get("conv_selector"):
        if st.session_state.get("chat_conv_id") in options:
            st.session_state["conv_selector"] = st.session_state["chat_conv_id"]

    # 陈旧夹紧：选择器残留的 id 若已不在列表（被删/换账号），先归 None 再渲染，
    # 否则 selectbox 拿着无效值实例化会越界报错（与文档列表分页夹紧同一手法）
    if st.session_state.get("conv_selector") not in options:
        st.session_state["conv_selector"] = None

    selected = st.selectbox(
        "当前对话", options, format_func=_fmt_conv, key="conv_selector"
    )

    # —— 切换检测：选择器变了 → 从服务端拉该会话消息覆盖本地缓存 ——
    if selected != st.session_state.get("chat_conv_id"):
        st.session_state.chat_conv_id = selected
        if selected is None:
            st.session_state.chat_history = []
        else:
            try:
                msgs = client.list_messages(selected)
            except ApiError as e:
                st.error(e.message)
                # 不在这里改 conv_selector（widget 已实例化）；把当前会话置空，
                # 下次重跑由上面的回填/夹紧逻辑把选择器归位，避免每轮循环报错
                st.session_state.chat_conv_id = None
                st.session_state.chat_history = []
            else:
                # 服务端消息 → 渲染缓存（字段口径与本地追加的消息一致，两条路径不打架）
                st.session_state.chat_history = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "sources": m.get("sources") or [],
                        "hit": bool(m.get("hit")),
                    }
                    for m in msgs
                ]

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
selected_groups = st.multiselect(
    "选择要检索的分组",
    list(name_to_id.keys()),
    default=list(name_to_id.keys()),
)
if not selected_groups:
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
    # 「清空」升级为「删除当前会话」：历史已落库，只清本地缓存的话刷新一下就回来了——
    # 想真正删必须删服务端会话（级联删消息），UI 承诺与实际行为必须一致
    if st.button("删除当前对话", icon=":material/delete:"):
        conv_id = st.session_state.get("chat_conv_id")
        try:
            if conv_id is not None:
                client.delete_conversation(conv_id)
        except ApiError as e:
            st.error(e.message)
            st.stop()
        # 本地状态归零：chat_conv_id 置 None，下次重跑由侧边栏回填逻辑把选择器归位
        st.session_state.chat_history = []
        st.session_state.chat_conv_id = None
        st.rerun()

# submit_mode="disable"：回答生成期间禁用输入框——防止连发把 7B 推理队列打成长龙
# （后端虽有互斥闸门兜底，前端体验上也不该让用户误以为「卡了」）
prompt = st.chat_input("就课件内容提问（支持术语/对比问法）", submit_mode="disable")
if prompt:
    if not selected_groups:
        st.warning("请先选择检索分组再提问。")
        st.stop()

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 首问建会话：标题=首问前 30 字（后端还会截到 128/兜底空标题，双端复校）。
        # 失败不回滚：刚建的空会话保留，它仍是当前会话，重问直接复用、不再重复建壳
        conv_id = st.session_state.get("chat_conv_id")
        if conv_id is None:
            try:
                conv = client.create_conversation(prompt[:30])
            except ApiError as e:
                st.error(e.message)
                st.stop()
            conv_id = conv["id"]
            # 只写 chat_conv_id（当前会话），conv_selector 留到下次重跑回填——
            # 此刻 widget 已实例化，直接改它的 key 会抛 StreamlitAPIException
            st.session_state.chat_conv_id = conv_id

        try:
            # group_ids 显式传选中分组（后端还会逐个校验归属，防越权检索）
            result = client.ask(
                prompt,
                group_ids=[name_to_id[n] for n in selected_groups],
                conversation_id=conv_id,
            )
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
            # 失败也进本地历史：会话脉络不断线，用户知道哪一轮出了问题。
            # 失败轮服务端没有落库（后端出错即未走到 append）——刷新后这轮消失属预期，重问即可。
            st.error(e.message)
            history.append({"role": "assistant", "content": f"请求失败：{e.message}", "sources": [], "hit": False})
