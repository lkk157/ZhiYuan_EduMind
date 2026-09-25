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
    """把 [{file_name, page_no, snippet, score}] 渲染成**默认折叠**的来源详情。

    为什么折叠（2026-09-25 产品反馈「来源太长影响体验」）：
    答案末尾的【来源：文件名，第X页】已是常驻溯源（卖点所在，一行），
    下面再平铺「标题 + 每来源一张带边框卡片 + 80 字片段」等于把同一件事
    说了三遍、3 个来源占半屏——常态是扫一眼页码，核对片段才是低频动作。
    于是：默认只占一个折叠条（1 行），点开才见详情；卡片边框也去掉，
    改成紧凑列表（嵌套边框在折叠层里就是纯浪费的内边距）。
    """
    if not sources:
        return
    with st.expander(f":material/link: 来源（{len(sources)} 项）"):
        for i, src in enumerate(sources, start=1):
            # 一行一来源：序号+文件+页码在前（溯源三要素），片段与相似度跟在破折号后
            line = f"**{i}. {src.get('file_name', '?')} · 第 {src.get('page_no', '?')} 页**"
            snippet = (src.get("snippet") or "").strip()
            score = src.get("score")
            meta = ""
            if snippet:
                meta += snippet
            if score is not None:
                meta += f"（相似度 {score:.2f}）" if meta else f"相似度 {score:.2f}"
            st.markdown(f"{line}  \n{meta}" if meta else line)


def render_hit_badge(hit: bool) -> None:
    """命中/兜底状态徽标。

    为什么未命中用 info 而不是 error：兜底不是错误，恰恰是「宁可不说也不编」的
    产品承诺——用中性醒目样式呈现，并把话术写成引导（换问法/传课件）。
    """
    if hit:
        st.caption(":material/check_circle: 已命中知识库")
    else:
        st.info("知识库中未找到相关内容——系统不会编造答案。可换一种问法，或先上传相关课件。")
