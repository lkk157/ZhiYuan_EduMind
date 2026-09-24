# -*- coding: utf-8 -*-
"""
图片解析：图片文件 = 一页空文本文档，真文本由 OCR 回填层（ingest/ocr.py）产出。

为什么图片不在这里直接 OCR：
解析层保持「纯本地、零模型调用」（pypdf/python-docx/python-pptx 同一约定）——
格式差异只存在于 parsers 包内，模型调用统一收口在 ocr.py（显存互斥序列只能有一处），
pipeline 的编排顺序也因此清晰：先解析（可能得到空页）→ 再 OCR 回填。
"""
from pathlib import Path

from app.ingest.parsers import ParsedPage

# 支持的图片后缀（与前端上传白名单、parse_document 分发保持一致）
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_image(path: Path) -> tuple[list[ParsedPage], list[int]]:
    """图片当一页处理：文本恒为空串、页码恒为 1，登记进 empty_pages 等 OCR 回填。

    为什么空页也保留在 pages 列表里：page_count 必须等于 1（前端展示页数），
    且 OCR 回填靠「empty_pages 里的页码 ↔ pages 里的页」对齐写回文本。
    """
    return [ParsedPage(page_no=1, text="")], [1]
