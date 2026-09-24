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


# ---------- 上次入库结果（先于操作区展示，重跑不丢；批量 = 逐文件一张卡片）----------
report = st.session_state.get("upload_report")
if report:
    for r in report["results"]:
        with st.container(border=True):
            if r["ok"]:
                st.markdown(f":material/task_alt: **{r['file_name']}** 入库完成（分组：{report['group']}）")
                if r.get("skipped_identical"):
                    # 文件级指纹短路：逐字节相同的重传直接跳过——增量的第一道闸
                    st.info("内容与上次上传完全一致，已跳过（文件级指纹短路，零重算）。")
                else:
                    st.markdown(
                        f"新增 **{r['added']}** 块 · 修改 **{r['changed']}** 块 · "
                        f"删除 **{r['removed']}** 块 · 未变 **{r['unchanged']}** 块"
                    )
                    st.caption(f"共 {r['page_count']} 页 / {r['chunk_count']} 块进入向量库")
                    if r.get("empty_pages"):
                        st.warning(
                            f"第 {_fmt_pages(r['empty_pages'])} 页未能识别出文本（空白页或 OCR 失败），"
                            "已登记；重新上传该文件可重试。"
                        )
            else:
                # 单文件失败不连坐整批：失败卡片独立呈现，其余文件照常入库
                st.markdown(f":material/error: **{r['file_name']}** 入库失败")
                st.error(r.get("error") or "未知错误")
    if report.get("failed"):
        st.caption(
            f"本批共 {len(report['results'])} 个文件：成功 {report['succeeded']}，"
            f"失败 {report['failed']}（失败文件可单独重传）"
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
# 上限数字与 .env 的 MAX_UPLOAD_FILES / MAX_UPLOAD_TOTAL_MB 对齐（服务端为准，超限会收到 400 人话提示）
st.caption(
    "支持 PDF / Word / PPT / 图片（png、jpg），可多选——单批最多 5 个、总大小不超过 200MB。"
    "同名重传 = 增量更新（只重算变化块）；含扫描页、图表或图片的文档会经 OCR 识别，耗时略长。"
)
with st.form("upload_form"):
    up = st.file_uploader(
        "选择课件文件（可多选）",
        type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.form_submit_button("上传并入库", icon=":material/upload:"):
        if not up:
            st.warning("请先选择文件。")
        else:
            try:
                # getvalue() 拿全量字节；逐文件组包交给后端批量接口（顺序处理、互不连坐）
                files = [(f.name, f.getvalue(), f.type) for f in up]
                resp = client.upload_documents(sel["id"], files)
                resp["group"] = sel_name  # 仅展示用的分组名装饰（服务端响应不含它）
                st.session_state.upload_report = resp
                st.rerun()  # 重跑后由顶部报告区展示逐文件结果（重跑不丢）
            except ApiError as e:
                st.error(e.message)  # 批次超限等 400 的人话提示从这里透出

# ---------- 文档列表 ----------
st.subheader(f"文档列表（{sel_name}）")
try:
    docs = client.list_documents(sel["id"])
except ApiError as e:
    st.error(e.message)
    st.stop()

if not docs:
    st.caption("该分组还没有文档。")

# ---------- 分页（产品需求 2026-09-24：每页最多 10 项，页码可选）----------
# 为什么在前端分页而不是改接口：个人知识库是几十上百的文档量级，列表一次返回即可；
# 生产环境量大时应改服务端分页（page/size + 索引）——又一处「毕设降级 vs 升级路径」。
DOCS_PAGE_SIZE = 10
if docs:
    total_pages = max(1, -(-len(docs) // DOCS_PAGE_SIZE))  # 向上取整
    page_key = f"docs_page_{sel['id']}"  # 分页状态按分组隔离：切组互不串页
    # 先夹紧再渲染：删除导致页数收缩时，session_state 残留的大页码会让 selectbox 越界
    if st.session_state.get(page_key, 1) > total_pages:
        st.session_state[page_key] = total_pages
    pager = st.container(horizontal=True)
    with pager:
        page_no = st.selectbox("页码", list(range(1, total_pages + 1)), key=page_key)
        st.caption(f"共 {len(docs)} 项 / {total_pages} 页")
    start = (page_no - 1) * DOCS_PAGE_SIZE
    docs = docs[start : start + DOCS_PAGE_SIZE]  # 只渲染当前页的 10 项

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
