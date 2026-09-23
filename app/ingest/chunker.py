# -*- coding: utf-8 -*-
"""
切块层：页感知滑窗，把逐页文本切成可入库的文本块。

为什么块绝不跨页（page_no 是溯源根基）：
最终答案引用的是 (file_name, page_no)——一旦某个块横跨两页，引用页码只能二选一，
另一页的内容就变成「查得到却说不出来源」；宁可多切几块，也不毁溯源锚点。
"""
from dataclasses import dataclass

from app.ingest.parsers import ParsedPage


@dataclass
class Chunk:
    """一个文本块：chunk_index 全文档连续（0 起），page_no 保留页级溯源。

    为什么 chunk_index 是「全档连续」而不是「页内连续」：
    向量 id = "{document_id}:{chunk_index}" 要求唯一且稳定，全档唯一号免去
    「页号+页内号」两段拼接的歧义，也让 diff 按单一维度对齐。
    """

    chunk_index: int
    page_no: int
    text: str


def chunk_pages(pages: list[ParsedPage], chunk_size: int, overlap: int) -> list[Chunk]:
    """页内滑窗切块：块绝不跨页、相邻块重叠 overlap 字符、chunk_index 全文档 0 起连续。

    为什么页内滑窗而不是全文连续切：
    全文切块会撕开页边界（溯源失效，见模块 docstring）；页内滑窗的步进 =
    chunk_size - overlap，相邻块保留 overlap 字符衔接语境，避免答案恰好卡在块边界上
    被腰斩（RAG 经典召回空洞）。

    为什么空文本页直接跳过：无文本层页没有可检索内容（占位文本会污染检索，
    取舍见 parsers 包 docstring），跳过后 chunk_index 仍连续，只是不含空页的块。
    """
    # 配置来自 settings（chunk_size / chunk_overlap），非法值宁可立刻炸出也不静默空转：
    # overlap >= chunk_size 会让步进 <= 0，切块循环永不前进
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        if not text.strip():
            continue  # 空文本页不产块（扫描页 / 纯图页）
        total = len(text)
        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            # 原文切片直接进指纹：不做 strip / 归一，保证 chunk_sha256 确定可复现
            chunks.append(
                Chunk(chunk_index=len(chunks), page_no=page.page_no, text=text[start:end])
            )
            if end >= total:
                break  # 已到页尾（含「整页不足一块」的短页）
            start += chunk_size - overlap  # 步进 = 块长 - 重叠，相邻块共享 overlap 字符
    return chunks
