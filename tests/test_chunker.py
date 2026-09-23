# -*- coding: utf-8 -*-
"""
切块层单测（M2）：页感知滑窗的核心承诺——块绝不跨页、重叠衔接、索引连续。

为什么这块必须测死：page_no 是溯源根基，一旦切块跨页，
答案末尾的【来源：文件名，第X页】就必然引用错页——卖点直接崩塌。
"""
import pytest

from app.ingest.chunker import chunk_pages
from app.ingest.parsers import ParsedPage


def test_short_page_yields_single_chunk():
    """短页（不足一块）：整页一块，页码原样，索引 0 起。"""
    chunks = chunk_pages([ParsedPage(page_no=1, text="hi")], chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "hi"
    assert chunks[0].page_no == 1
    assert chunks[0].chunk_index == 0


def test_long_page_sliding_window_with_overlap():
    """长页滑窗：相邻块共享 overlap 字符衔接语境，避免答案卡在块边界被腰斩。"""
    text = "a" * 8 + "b" * 8 + "c" * 9  # 25 字符
    chunks = chunk_pages([ParsedPage(page_no=3, text=text)], chunk_size=10, overlap=2)
    assert len(chunks) == 3  # [0:10] [8:18] [16:25]
    assert chunks[0].text == text[0:10]
    assert chunks[1].text == text[8:18]
    assert chunks[2].text == text[16:25]
    # 重叠区：后块开头 == 前块结尾的 overlap 个字符
    assert chunks[1].text[:2] == chunks[0].text[-2:]
    assert all(c.page_no == 3 for c in chunks)


def test_chunks_never_cross_page_boundary():
    """多页文档：每个块只属于一页（绝不跨页），chunk_index 全文档连续。"""
    pages = [ParsedPage(page_no=1, text="a" * 25), ParsedPage(page_no=2, text="b" * 5)]
    chunks = chunk_pages(pages, chunk_size=10, overlap=0)
    assert [c.page_no for c in chunks] == [1, 1, 1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    # 跨页防线：第一页的块里绝不出现第二页的字符
    assert all("b" not in c.text for c in chunks[:3])


def test_empty_and_blank_pages_skipped():
    """空文本页/纯空白页不产块（占位文本会污染检索），索引仍连续。"""
    pages = [
        ParsedPage(page_no=1, text=""),
        ParsedPage(page_no=2, text="   \n "),
        ParsedPage(page_no=3, text="hello"),
    ]
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].page_no == 3
    assert chunks[0].chunk_index == 0


def test_chunk_text_keeps_raw_slice():
    """原文切片直接进指纹：不 strip 不归一，chunk_sha256 才能确定可复现。"""
    chunks = chunk_pages([ParsedPage(page_no=1, text="  padded  ")], chunk_size=100, overlap=0)
    assert chunks[0].text == "  padded  "


def test_invalid_params_raise():
    """非法参数宁可立刻炸出也不静默空转（overlap >= size 会让切块循环永不前进）。"""
    page = [ParsedPage(page_no=1, text="x" * 50)]
    with pytest.raises(ValueError):
        chunk_pages(page, chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_pages(page, chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_pages(page, chunk_size=10, overlap=-1)
