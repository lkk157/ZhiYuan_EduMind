# -*- coding: utf-8 -*-
"""
前端入口：登录/注册门 + st.navigation 页面路由（streamlit run frontend/app.py）。

两个关键设计（都写「为什么」）：
1. 登录门做在入口、页面按登录态**动态构建**——官方推荐的「条件页面」模式：
   未登录时 navigation 列表里根本没有「上传管理/智能答疑」两个路由，从导航层就进不去；
   页面脚本里再各放一道鉴权守卫是纵深防御第二层（防直接刷新/旧链接进入）。
2. 为什么页面目录叫 app_pages/ 而不是 pages/：Streamlit 1.5x+ 官方多页面文档明确警告
   pages/ 会与旧版自动发现 API 冲突导致意外行为，必须用 app_pages/ + st.navigation
   （本项目 ROADMAP 原计划写的是 pages/，此处按框架版本要求调整，功能文件名不变）。

为什么 token 放 session_state：每个浏览器标签页一个会话、互不串号；
前端只保管令牌不发明身份——身份真伪由后端 JWT + user_id 回查保证（M1 的 get_current_user）。
"""
import streamlit as st

from services.api import API_BASE_URL, ApiClient, ApiError

# 页面全局配置（必须是第一个 st 调用）；图标用 Material Symbols 代码而非 emoji——
# 既符合官方「少用 emoji」规范，也避开 Windows GBK 控制台打印崩溃的老坑（风险 R4）
st.set_page_config(page_title="知源 ZhiYuan", page_icon=":material/school:", layout="centered")

# ---- 会话状态集中初始化（一处 setdefault，避免散落赋值引发 KeyError / 串号）----
st.session_state.setdefault("token", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("chat_history", [])  # 答疑页当前会话的消息缓存（服务端 MySQL 为事实源）
st.session_state.setdefault("chat_conv_id", None)  # 当前会话 id（None=新对话，首问时才创建）
st.session_state.setdefault("upload_report", None)  # 最近一次入库结果（跨一次重跑展示）


def _reset_session() -> None:
    """退出/切换账号时清空全部会话态——不留上一个用户的聊天记录（隐私底线）。

    conv_selector 是会话选择器 widget 的 session key，必须 pop 而不是赋 None：
    留着上一个用户的会话 id，换人登录后选择器会拿着别人的 id 打接口（404 串号噪音）。
    """
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.chat_history = []
    st.session_state.chat_conv_id = None
    st.session_state.pop("conv_selector", None)
    st.session_state.upload_report = None


# ================= 未登录：登录/注册门 =================
if not st.session_state.token:
    st.title("知源 ZhiYuan", icon=":material/school:")
    st.caption("基于多模态 RAG 的高校学科教育智能答疑系统")

    # segmented_control 是官方推荐的「二选一」控件（优于横排 radio）
    mode = st.segmented_control("模式", ["登录", "注册"], label_visibility="collapsed") or "登录"

    with st.form("auth_form", clear_on_submit=True):
        username = st.text_input("用户名", placeholder="至少 3 个字符")
        password = st.text_input("密码", type="password", placeholder="至少 6 个字符")
        submitted = st.form_submit_button(mode, width="stretch", icon=":material/login:")

    if submitted:
        client = ApiClient()
        try:
            if mode == "注册":
                client.register(username.strip(), password)
                st.success("注册成功，请切换到「登录」。")
            else:
                info = client.login(username.strip(), password)
                st.session_state.token = info["access_token"]
                st.session_state.username = info.get("username", username.strip())
                st.rerun()  # 进入已登录导航
        except ApiError as e:
            # 后端统一出口的人话提示直接透出（如「用户名或密码错误」）
            st.error(e.message)

    st.caption(f"后端地址：{API_BASE_URL}（可用环境变量 API_BASE_URL 覆盖）")

# ================= 已登录：侧边栏导航 =================
else:
    with st.sidebar:
        st.markdown(f":material/person: **{st.session_state.username}**")
        if st.button("退出登录", icon=":material/logout:", width="stretch"):
            _reset_session()
            st.rerun()

    pages = [
        st.Page("app_pages/1_上传管理.py", title="上传管理", icon=":material/upload_file:"),
        st.Page("app_pages/2_智能答疑.py", title="智能答疑", icon=":material/school:"),
        st.Page("app_pages/3_错题与学情.py", title="错题与学情", icon=":material/insights:"),
        st.Page("app_pages/4_数据看板.py", title="数据看板", icon=":material/dashboard:"),
    ]
    page = st.navigation(pages, position="sidebar")
    # 标题统一在入口处理（官方模式：页面内不再重复 st.title）
    st.title(page.title, icon=page.icon)
    page.run()
