# -*- coding: utf-8 -*-
"""
上传管理页：分组 CRUD + 课件上传（增量入库结果可视化）+ 文档列表。

为什么入库结果要展示 added/changed/removed 计数：这是「增量向量化」的可视化证据
（生产思维 #5 的毕设落点）——同名重传时用户能直接看到「内容没变就跳过、
改了几页只重算几块」，答辩演示增量就指这一栏。

为什么 empty_pages 要明示：扫描页/纯图页「跳过但登记」是 M2 的既定取舍
（OCR 在多模态阶段接管）——不展示就等于把限制藏起来，展示了就是「有账、有预案」。
"""
import streamlit as st

from services.api import ApiClient, ApiError

# 鉴权守卫（纵深防御第二层：导航已按登录态隐藏，这里防旧链接/刷新直入）
if not st.session_state.get("token"):
    st.warning("请先登录后再使用知识库功能。")
    st.stop()

client = ApiClient(token=st.session_state.token)


def _fmt_pages(pages: list) -> str:
    """页码列表 → 人话（如「3、7、12」）。"""
    return "、".join(str(p) for p in pages)


# ---------- 上次入库结果（先于操作区展示，重跑不丢）----------
report = st.session_state.get("upload_report")
if report:
    with st.container(border=True):
        st.markdown(f":material/task_alt: **{report['file_name']}** 入库完成（分组：{report['group']}）")
        if report.get("skipped_identical"):
            # 文件级指纹短路：逐字节相同的重传直接跳过——增量的第一道闸
            st.info("内容与上次上传完全一致，已跳过（文件级指纹短路，零重算）。")
        else:
            st.markdown(
                f"新增 **{report['added']}** 块 · 修改 **{report['changed']}** 块 · "
                f"删除 **{report['removed']}** 块 · 未变 **{report['unchanged']}** 块"
            )
            st.caption(f"共 {report['page_count']} 页 / {report['chunk_count']} 块进入向量库")
            if report.get("empty_pages"):
                st.warning(
                    f"第 {_fmt_pages(report['empty_pages'])} 页未能识别出文本（空白页或 OCR 失败），"
                    "已登记；重新上传该文件可重试。"
                )
        if st.button("关闭", key="close_report"):
            st.session_state.upload_report = None
            st.rerun()

# ---------- 分组管理 ----------
st.subheader("知识库分组")
st.caption("按学科/课程分组隔离——不同分组的向量库物理独立，检索互不串扰。")

with st.form("create_group_form", clear_on_submit=True):
    new_name = st.text_input("分组名称", placeholder="如：高等数学 / 机器学习导论")
    if st.form_submit_button("新建分组", icon=":material/add:"):
        name = new_name.strip()
        if not name:
            st.warning("分组名不能为空。")
        else:
            try:
                client.create_group(name)
                st.rerun()
            except ApiError as e:
                st.error(e.message)

try:
    groups = client.list_groups()
except ApiError as e:
    st.error(e.message)
    st.stop()

if not groups:
    st.info("还没有分组。请先新建一个分组，再上传课件。")
    st.stop()

# 当前分组选择 + 删除（同名分组在后端已有唯一约束，name→id 映射安全）
name_to_group = {g["name"]: g for g in groups}
sel_name = st.selectbox("当前分组", list(name_to_group.keys()))
sel = name_to_group[sel_name]

with st.popover("删除当前分组"):
    # 删除是不可恢复操作（连带文档/向量/文件），必须二次确认（官方交互规范）
    st.warning(f"将删除分组「{sel_name}」及其全部文档与向量，不可恢复。")
    if st.button("确认删除", key="confirm_del_group", icon=":material/delete:"):
        try:
            client.delete_group(sel["id"])
            st.session_state.upload_report = None
            st.rerun()
        except ApiError as e:
            st.error(e.message)

# ---------- 上传 ----------
st.subheader("上传课件")
st.caption(
    "支持 PDF / Word / PPT / 图片（png、jpg）。同名重传 = 增量更新（只重算变化块）；"
    "含扫描页、图表或图片的文档会经 OCR 识别，耗时略长。"
)
with st.form("upload_form"):
    up = st.file_uploader("选择课件文件", type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"])
    if st.form_submit_button("上传并入库", icon=":material/upload:"):
        if up is None:
            st.warning("请先选择文件。")
        else:
            try:
                # getvalue() 拿全量字节（二进制容器绝不能走文本读写）
                result = client.upload_document(sel["id"], up.name, up.getvalue())
                result["file_name"] = up.name
                result["group"] = sel_name
                st.session_state.upload_report = result
                st.rerun()  # 重跑后由顶部报告区展示结果（重跑不丢）
            except ApiError as e:
                st.error(e.message)

# ---------- 文档列表 ----------
st.subheader(f"文档列表（{sel_name}）")
try:
    docs = client.list_documents(sel["id"])
except ApiError as e:
    st.error(e.message)
    st.stop()

if not docs:
    st.caption("该分组还没有文档。")

for d in docs:
    with st.container(border=True):
        row = st.container(horizontal=True)
        with row:
            st.markdown(f":material/description: **{d['file_name']}**")
            if st.button("删除", key=f"del_doc_{d['id']}", icon=":material/delete:"):
                try:
                    client.delete_document(d["id"])
                    st.rerun()
                except ApiError as e:
                    st.error(e.message)
        st.caption(f"状态 {d['status']} · {d['page_count']} 页 · {d['chunk_count']} 块")
        if d.get("empty_pages"):
            st.caption(
                f":material/scanner: 第 {_fmt_pages(d['empty_pages'])} 页未识别出文本（空白/失败），重新上传可重试"
            )
