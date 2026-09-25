# -*- coding: utf-8 -*-
"""
错题与学情页（M5）：错题本（含知识点图谱关联推荐）+ 学习周报 + 记忆清单。

为什么错题与周报放同一页：两者是同一数据链的两端——错题是「原始信号」，
周报是「提炼结论」，放一起答辩时的故事线完整：判错 → 记录 → 关联推荐 → 周报点名。

关联推荐的「去问一问」：写 pending_prompt 后 st.switch_page 跳答疑页——
与示例问题同一注入机制（不另写提问逻辑，两条路必然同构）。
"""
import streamlit as st

from services.api import ApiClient, ApiError

# 鉴权守卫（纵深防御第二层，与其余页面同款）
if not st.session_state.get("token"):
    st.warning("请先登录后再使用错题与学情。")
    st.stop()

client = ApiClient(token=st.session_state.token)

# ---------- 学习周报 ----------
st.subheader("学习周报")
try:
    latest = client.latest_report()
except ApiError as e:
    st.error(e.message)
    latest = {"report": None}

if latest.get("report"):
    st.caption(f"最近生成：{latest.get('created_at') or '（时间未知）'}")
    st.markdown(latest["report"])
else:
    st.info("还没有生成过周报——生成后会点名薄弱知识点并给出复习建议。")

col_gen, col_note = st.columns([1, 2])
with col_gen:
    if st.button("生成本周学情报告", icon=":material/refresh:", width="stretch"):
        with st.spinner("正在分析近期对话与答题记录（约 10–30 秒）…"):
            try:
                report = client.generate_report()
                st.session_state["last_report"] = report
                st.rerun()
            except ApiError as e:
                st.error(e.message)
with col_note:
    st.caption(
        "报告由 LLM 依据你最近的对话与答题记录提炼（每生成一次会写入记忆，"
        "并自动注入之后的答疑——因材施教的闭环）。"
    )

# ---------- 错题本（含知识点图谱关联推荐）----------
st.subheader("错题本")
try:
    records = client.quiz_records()
except ApiError as e:
    st.error(e.message)
    records = []

if not records:
    st.caption("还没有错题——出题作答判错后会自动记录在这里。")

for rec in records:
    with st.container(border=True):
        row = st.container(horizontal=True)
        with row:
            st.markdown(f":material/error: **{rec['question']}**")
            if st.button("删除", key=f"del_rec_{rec['id']}", icon=":material/delete:"):
                try:
                    client.delete_quiz_record(rec["id"])
                    st.rerun()
                except ApiError as e:
                    st.error(e.message)
        my_answer = rec.get("user_answer") or "（未作答）"
        st.markdown(f"**我的作答**：{my_answer}")
        st.markdown(f"**正确答案**：{rec.get('correct_answer') or '—'}")
        if rec.get("explanation"):
            st.caption(f"解析：{rec['explanation']}")
        if rec.get("ref_file"):
            st.caption(f"教材出处：{rec['ref_file']} 第 {rec.get('ref_page') or '?'} 页")

        # 知识点图谱关联推荐（动态现算：新上传的课件自动入图）
        related = rec.get("related") or []
        if related:
            st.markdown(":material/account_tree: **关联知识点（图谱推荐）**")
            for rel in related:
                rel_col, btn_col = st.columns([3, 1])
                with rel_col:
                    score_txt = f"（相似度 {rel['score']:.2f}）" if rel.get("score") is not None else ""
                    st.markdown(f"- **{rel['label']}** — {rel['relation']} {score_txt}")
                with btn_col:
                    if st.button(
                        "去问一问",
                        key=f"ask_rel_{rec['id']}_{rel['file_name']}",
                        icon=":material/chat:",
                    ):
                        # 跳答疑页并注入追问（与示例问题同一 pending_prompt 机制）
                        st.session_state["pending_prompt"] = f"请讲解一下「{rel['label']}」，并说明它和前面知识的联系"
                        st.switch_page("app_pages/2_智能答疑.py")

# ---------- 记忆清单（个性化「看得见」）----------
with st.expander("系统记住了关于我的这些（记忆清单）"):
    try:
        facts = client.memory_facts()
    except ApiError as e:
        st.error(e.message)
        facts = []
    if not facts:
        st.caption("暂无记忆——判错或生成周报后这里会出现条目。")
    for fact in facts:
        kind_label = {"weak_point": "薄弱点", "insight": "洞察", "report": "周报"}.get(
            fact["kind"], fact["kind"]
        )
        st.markdown(f"- **[{kind_label}]** {fact['content'][:120]}")
