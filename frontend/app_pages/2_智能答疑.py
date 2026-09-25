# -*- coding: utf-8 -*-
"""
智能答疑页：选分组提问 → 带溯源作答（会话历史持久化 + M4 Agent 意图路由）。

会话历史设计（2026-09-24 起）：
- MySQL 是事实源：每个会话的问答经 /chat/ask?conversation_id= 落库，
  刷新页面/换设备重新登录都能回看（旧版只存 session_state，一刷就丢）；
- 本页 session_state.chat_history 只是「当前会话的渲染缓存」——
  切换会话时从服务端整拉覆盖，新建会话时清空，任何时候以服务端为准；
- 「＋ 新对话」选择哨兵：首问时才真正建会话行（标题=首问截 30 字，零 LLM 调用），
  避免点一下「新建」就留一个永远空着的会话壳。

M4 增强：
- 意图徽章（答疑/出题/总结/计算）让 Agent 路由「看得见」——答辩演示指这里；
  回看历史时 intent 不落库（message 表无该列，加列需迁移），按内容特征识别试题，
  其余意图回看不显示徽章（活体显示、回看降级，见 components/quiz.try_parse_quiz 注释）；
- 引导式答疑开关（苏格拉底模式）：只影响下一次提问的 system prompt；
- 试题 JSON 的消息渲染成可作答卡片，回看时同一张卡可再次作答判分。

为什么未命中样式是 info 不是 error：兜底不是出错，而是「宁可不说也不编」的产品承诺
（后端空检索分支根本不会调用 LLM）——UI 口径与后端卖点保持一致。
"""
import streamlit as st

from components.quiz import render_quiz_card, try_parse_quiz
from components.sources import render_hit_badge, render_sources
from services.api import ApiClient, ApiError

# 意图标签中文映射（与后端 intent.VALID_INTENTS 对齐，只做展示）
_INTENT_LABELS = {"qa": "答疑", "quiz": "出题", "summary": "总结", "calc": "计算"}

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

    # —— 选择器回填（必须在 selectbox 实例化之前消费；实例化后改 key 会抛异常）——
    # ★踩坑记录（2026-09-24 实测）：最初这里写的是「chat_conv_id != conv_selector
    # 就把选择器拉回 chat_conv_id」，结果是灾难——Streamlit 用户点选 selectbox 时
    # 会先把新值写进 session_state 再重跑脚本，这段「对齐」把用户的每次点选都在
    # 脚本开头弹回原值：点「＋ 新对话」弹回旧会话（新建不了）、登录后点旧对话弹回
    # 哨兵（回不去）。根因：「值不等」分不清是【用户点选】还是【主区新建/删除】两种方向。
    # 修法：方向必须显式——只有主区（新建会话处）写 conv_selector_pending 信号，
    # 侧边栏只消费信号、绝不主动猜测对齐；用户点选由切换检测分支自己处理。
    if "conv_selector_pending" in st.session_state:
        st.session_state["conv_selector"] = st.session_state.pop("conv_selector_pending")

    # 首次进入本页（登录后 conv_selector 尚未存在）：默认选中最近活跃会话——
    # 「重新登录还能回到旧对话」的产品语义就落在这里；无会话则停留在新对话哨兵。
    # （放在 pending 消费之后：新建流程里 key 已存在，不会被误覆盖）
    if "conv_selector" not in st.session_state and conversations:
        st.session_state["conv_selector"] = conversations[0]["id"]

    # 陈旧夹紧：选择器残留的 id 若已不在列表（被删/换账号），先归 None 再渲染，
    # 否则 selectbox 拿着无效值实例化会越界报错（与文档列表分页夹紧同一手法）。
    # 删除当前会话后的归位也由它兜底：旧 id 已不在列表 → 夹回新对话哨兵。
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
                # 不在这里改 conv_selector（widget 已实例化）；把当前会话置空——
                # 若该会话已不在列表，下次重跑由「陈旧夹紧」归位到新对话哨兵；
                # 若只是瞬时网络错（会话还在），下次重跑会自然重试加载
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
st.caption(
    "范围问法（质量优化）：「总结第1-10页」「第一章讲了什么」——"
    "页码直接过滤，章节按 PDF 目录映射（无目录的课件请用页码问法）。"
)

# 引导式答疑（M4 苏格拉底模式）：默认关——不影响既有直答行为，演示时现场打开
st.toggle(
    "引导式答疑（苏格拉底模式）",
    key="guide_mode",
    help="打开后不直接给完整答案，先反问引导你思考、再点到关键结论。",
)

# ---------- 对话区 ----------
st.subheader("对话")
history = st.session_state.chat_history
for idx, msg in enumerate(history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # 试题消息 → 渲染成交互卡片（活体与回看同一渲染路径，key 按序号隔离）
            quiz = try_parse_quiz(msg.get("content", ""))
            if quiz:
                render_quiz_card(quiz, key_prefix=f"quiz_{idx}", client=client)
                render_sources(msg.get("sources") or [], client=client, key_prefix=f"src_{idx}")
                continue
            st.markdown(msg["content"])
            intent = msg.get("intent")  # 活体回答才有（回看从服务端拉，无 intent 列）
            if intent == "calc":
                # 计算工具的「命中」语义与知识库不同（无来源），不显示命中徽章，
                # 改显意图标签，避免「已命中知识库」配上空来源的自相矛盾
                st.caption(f":material/route: 意图：{_INTENT_LABELS.get(intent, intent)}")
            else:
                if intent:
                    st.caption(f":material/route: 意图：{_INTENT_LABELS.get(intent, intent)}")
                render_hit_badge(msg.get("hit", True))
            render_sources(msg.get("sources") or [], client=client, key_prefix=f"src_{idx}")
            # 范围解析降级提示（如「课件无目录，请用第X-Y页问法」）——
            # 只有活体回答带 scope_note（服务端不落库，回看不显示；见模块注释）
            if msg.get("scope_note"):
                st.info(msg["scope_note"])
        else:
            st.markdown(msg["content"])


def _history_to_markdown(items: list[dict]) -> str:
    """当前对话 → 复习笔记 Markdown（体验增强包：会话导出）。

    纯前端拼装（对话就在 session_state，零后端改动）；来源逐条列出，
    导出的笔记保留「可核对」这一核心价值，而不只是问答文本。
    """
    lines = ["# 知源 · 问答笔记", ""]
    for m in items:
        if m["role"] == "user":
            lines += ["## 提问", "", m.get("content", ""), ""]
        else:
            lines += ["## 回答", "", m.get("content", ""), ""]
            for s in m.get("sources") or []:
                score = s.get("score")
                score_txt = f"（相似度 {score:.2f}）" if score is not None else ""
                lines.append(f"- {s.get('file_name', '?')} 第 {s.get('page_no', '?')} 页{score_txt}")
            if m.get("sources"):
                lines.append("")
    return "\n".join(lines)


if history:
    # 删除会话（真正删服务端，UI 承诺与行为一致）+ 导出笔记（当前会话内容下载）
    col_del, col_export = st.columns(2)
    with col_del:
        if st.button("删除当前对话", icon=":material/delete:", width="stretch"):
            conv_id = st.session_state.get("chat_conv_id")
            try:
                if conv_id is not None:
                    client.delete_conversation(conv_id)
            except ApiError as e:
                st.error(e.message)
                st.stop()
            # 本地状态归零；选择器归位交给下次重跑的「陈旧夹紧」——
            # 旧 id 已从列表消失，夹紧逻辑会把它带回新对话哨兵（无需再写 pending）
            st.session_state.chat_history = []
            st.session_state.chat_conv_id = None
            st.rerun()
    with col_export:
        st.download_button(
            "导出笔记 (.md)",
            data=_history_to_markdown(history),
            file_name="知源问答笔记.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )

# 示例问题（空对话冷启动）：点击写 pending_prompt，与 chat_input 走同一条提问路径——
# 不另写处理逻辑，避免两条路径漂移（体验增强包 #4）
if not history:
    st.caption("不知道问什么？试试：")
    ex_cols = st.columns(3)
    for ex_col, example in zip(
        ex_cols, ["这个知识库讲了什么？", "总结第1-5页", "出3道随堂测"]
    ):
        with ex_col:
            if st.button(example, key=f"exq_{example}", width="stretch"):
                st.session_state["pending_prompt"] = example
                st.rerun()

# submit_mode="disable"：回答生成期间禁用输入框——防止连发把 7B 推理队列打成长龙
# （后端虽有互斥闸门兜底，前端体验上也不该让用户误以为「卡了」）
prompt = st.chat_input(
    "提问 / 出题（如：出3道题）/ 总结上一节 / 纯算式计算",
    submit_mode="disable",
)
prompt = prompt or st.session_state.pop("pending_prompt", None)  # 示例问题注入
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
            # 此刻 selectbox 已实例化，不能直接改 conv_selector（会抛异常）——
            # 写 pending 信号，由下一次重跑的侧边栏在实例化前消费回填（见侧边栏注释）
            st.session_state.chat_conv_id = conv_id
            st.session_state["conv_selector_pending"] = conv_id

        try:
            final: dict = {}

            def _tokens():
                """SSE 事件 → 打字机增量；done 存入 final、error 转统一 ApiError。"""
                for event in client.ask_stream(
                    prompt,
                    group_ids=[name_to_id[n] for n in selected_groups],
                    conversation_id=conv_id,
                    guide_mode=bool(st.session_state.get("guide_mode")),
                ):
                    kind = event.get("t")
                    if kind == "delta":
                        yield event.get("v") or ""
                    elif kind == "done":
                        final.update(event)
                    elif kind == "error":
                        raise ApiError(int(event.get("code", 500)), event.get("message") or "生成失败")

            # 打字机逐字上屏：qa/总结有 delta；出题/计算无 delta（write_stream 立即返回），
            # done 后由重跑渲染循环出卡片/结果——所有分支最终以 done（净化后契约）为准。
            # 已知小瑕疵：若资料不足闸门触发，流上屏的是模型原始「无法回答…」开头文本，
            # 重跑后替换为兜底话术（两者语义一致，仅文案归一）。
            st.write_stream(_tokens)
            if not final:
                raise ApiError(500, "流式响应异常中断（未收到结果）")

            # 落历史缓存（intent/scope_note 只活在本会话内存里，回看时服务端没有该字段——见模块注释）
            history.append(
                {
                    "role": "assistant",
                    "content": final["answer"],
                    "sources": final.get("sources") or [],
                    "hit": final["hit"],
                    "intent": final.get("intent"),
                    "scope_note": final.get("scope_note"),
                }
            )
            # 重跑后由上方统一渲染循环上屏（成功/失败两条路径同构，避免两处逻辑漂移）
            st.rerun()
        except ApiError as e:
            # 失败也进本地历史：会话脉络不断线，用户知道哪一轮出了问题。
            # 失败轮服务端没有落库（后端出错即未走到 append）——刷新后这轮消失属预期，重问即可。
            history.append(
                {"role": "assistant", "content": f"请求失败：{e.message}", "sources": [], "hit": False}
            )
            st.rerun()
