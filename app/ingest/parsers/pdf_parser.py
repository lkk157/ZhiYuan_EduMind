# -*- coding: utf-8 -*-
"""
PDF 解析：pypdf 逐页抽取文本层；空页 / 无文本层页只登记页码，不产出任何文本。

为什么 M2 只吃文本层、不做 OCR：
OCR（GLM-OCR）是显存大户，按 CLAUDE.md §3 硬件红线必须与 LLM 时段隔离、用完即卸，
只能走 M4 的入库后处理流水线；M2 先把文本层吃干净，扫描页记入 empty_pages 留清单。
"""
from pathlib import Path

from pypdf import PdfReader

from app.ingest.parsers import ParsedPage


def parse_pdf(path: Path) -> tuple[list[ParsedPage], list[int]]:
    """逐页 extract_text，返回 (全部页, 无文本层页码列表)，页码 = PDF 真实页码（1 起）。

    为什么不给空页编占位文本（如「[扫描页]」）送进向量库——这是防污染检索的取舍：
    占位文本与任何提问都算不出真实语义相似度，却会以「合法块」的身份参与检索排序，
    轻则挤掉有效结果，重则被 LLM 当成真资料引用编造答案。所以空页 text 置空串、
    页码记入 empty_pages，向量库里绝不出现无信息块。

    M4 OCR 接管路径：读 empty_pages 清单逐页识别，把真文本回填成新块再增量入库，
    复用同一套 chunk / 指纹 / diff 流水线，本函数与切块层均无需改动。
    """
    reader = PdfReader(str(path))
    pages: list[ParsedPage] = []
    empty_pages: list[int] = []
    for page_no, page in enumerate(reader.pages, start=1):
        # pypdf 对无文本层页返回空串或纯空白，两种都算「无文本层」
        text = page.extract_text() or ""
        if not text.strip():
            empty_pages.append(page_no)
            # 空白归一成空串：下游判断「空文本页」只需看 not text.strip()，不留模糊地带
            text = ""
        pages.append(ParsedPage(page_no=page_no, text=text))
    return pages, empty_pages
