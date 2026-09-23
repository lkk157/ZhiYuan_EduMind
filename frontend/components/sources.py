# -*- coding: utf-8 -*-
"""
溯源展示组件：来源卡片 + 命中/兜底徽标。

为什么单独成组件：来源展示是本项目的「卖点 UI」——回答末尾【来源：文件名，第X页】的
明细化展开（文件名/页码/片段三要素），答疑页与后续题库页都要复用；
集中一处保证展示口径永远与答案末尾的【来源】一致（后端已保证同序同去重）。

为什么用原生容器而不是自绘 HTML 卡片：Streamlit 官方最佳实践优先原生元素
（st.container(border=True)），主题适配/移动端折叠都免费拿到，也不引入 XSS 面。
"""
import streamlit as st


def render_sources(sources: list[dict]) -> None:
    """把 [{file_name, page_no, snippet}] 渲染成来源卡片列表（与答案末尾【来源】同序）。"""
    if not sources:
        return
    st.markdown(":material/link: **来源**")
    for i, src in enumerate(sources, start=1):
        with st.container(border=True):
            # 序号 + 文件名 + 页码：溯源三要素一眼可见（答辩演示就指这里）
            st.markdown(f"**{i}. {src.get('file_name', '?')} · 第 {src.get('page_no', '?')} 页**")
            # 片段是「让用户核对出处」的证据，用 caption 弱化不抢答案的视觉焦点
            st.caption(src.get("snippet", ""))


def render_hit_badge(hit: bool) -> None:
    """命中/兜底状态徽标。

    为什么未命中用 info 而不是 error：兜底不是错误，恰恰是「宁可不说也不编」的
    产品承诺——用中性醒目样式呈现，并把话术写成引导（换问法/传课件）。
    """
    if hit:
        st.caption(":material/check_circle: 已命中知识库")
    else:
        st.info("知识库中未找到相关内容——系统不会编造答案。可换一种问法，或先上传相关课件。")
