# -*- coding: utf-8 -*-
"""
Word 解析：python-docx 按段落累积出「逻辑页」，为溯源提供确定性页码锚点。

为什么不用 Word 的「真实页码」，详见 parse_docx 的 docstring——一句话：
.docx 不存页码，页边界随渲染环境漂移，而逻辑页是纯内容函数，永远可复现。
"""
from pathlib import Path

from docx import Document as DocxDocument

from app.ingest.parsers import ParsedPage

# 逻辑页目标长度（字符）。
# 为什么是模块常量而不是 settings 配置：1000 是「逻辑页」契约语义的一部分
# （段落累积约 1000 字符切一页），属于口径而非可调阈值；真正随环境变的
# RAG 参数（chunk_size / top_k / score_threshold）才进 .env（CLAUDE.md §5）。
LOGICAL_PAGE_CHARS = 1000


def parse_docx(path: Path) -> tuple[list[ParsedPage], list[int]]:
    """段落累积约 1000 字符切一个逻辑页（单个段落绝不拆开），返回 (逻辑页列表, 空页列表恒为 [])。

    为什么 Word 无稳定分页（分页随渲染环境变化），用确定性逻辑页保证溯源锚点可复现：
    .docx 文件里根本不存页码——页边界由字体 / 纸张 / 打印机驱动在渲染时才决定，
    同一份文件在答辩机和开发机上可能差出一页，「第 X 页」的引用会随机漂移。
    按段落累积切逻辑页是纯内容函数：同一文件永远得到同一组 (page_no, text)，
    重传增量比对、答案引用页码、单测断言全部可复现。

    为什么段落绝不拆开：段落是语义单元，撕开的半个段落进向量库会让问答引用出半句话；
    代价是单段超长时该逻辑页超过 1000 字符（「约 1000」是软目标，整段完整是硬约束）。

    为什么空段落（排版留白、纯图片锚点）直接跳过：它们不占逻辑页字符预算，
    也不产出空白页——Word 的「图片落在第几页」M2 无从得知，交给 M4 OCR 时按整篇处理。
    """
    doc = DocxDocument(str(path))
    pages: list[ParsedPage] = []
    buf: list[str] = []  # 当前逻辑页的段落文本
    buf_len = 0  # 当前逻辑页已累积字符数（不含换行，约 1000 是软目标不必抠精确）

    def _flush() -> None:
        """把缓冲段落落成一个逻辑页（页号 = 已产出页数 + 1，保证 1 起连续）。"""
        nonlocal buf, buf_len
        pages.append(ParsedPage(page_no=len(pages) + 1, text="\n".join(buf)))
        buf = []
        buf_len = 0

    for para in doc.paragraphs:
        text = para.text  # 原文进指纹：不做 strip/大小写归一，重传同文本必同哈希
        if not text.strip():
            continue  # 空段落只贡献排版空白，跳过
        # 段落绝不拆开：整段放不下就先切页，宁可本页略短也不撕开段落
        if buf and buf_len + len(text) > LOGICAL_PAGE_CHARS:
            _flush()
        buf.append(text)
        buf_len += len(text)
    if buf:
        _flush()  # 尾部段落落成最后一页，防止丢失
    # 逻辑页天然非空（空段落已跳过），empty_pages 恒为空列表
    return pages, []
