# -*- coding: utf-8 -*-
"""
文档解析层：把 PDF / Word / PPT 统一成「逐页文本 + 无文本层页码」两个产物。

为什么统一成 ParsedPage 列表而不是各格式私有结构：
下游（切块/指纹/入库）必须完全不感知格式差异——page_no 是溯源锚点的唯一载体
（最终答案末尾【来源：文件名，第X页】全靠它），格式差异只允许存在于本包内。

为什么无文本层页码要单独作为第二个返回值：
空页（扫描版 PDF、纯图片幻灯片）目前没有任何可入库的文本，把占位文案
（如「[图片页]」）塞进向量库只会污染检索；单独列出页码，M4 OCR 接管时
按这份清单逐页识别回填——M2 与 M4 的交接点就在这里。
"""
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AppError


@dataclass
class ParsedPage:
    """一页解析结果：page_no 是溯源锚点，text 为该页全部文本（无文本层页恒为空串）。

    为什么无文本层页也要出现在列表里（而不是直接丢掉）：
    page_count 必须等于文档真实页数（含空页），前端展示页数与 M4 OCR 页码清单都依赖它；
    text 置空串即可——切块层会跳过空文本页，占位文本不会进向量库。
    """

    page_no: int
    text: str


def parse_document(path: Path) -> tuple[list[ParsedPage], list[int]]:
    """按文件后缀分发到对应解析器，返回 (逐页文本, 无文本层页码列表)。

    页码约定（层间契约，溯源锚点的地基）：
    - PDF  = 真实页码（1 起）；
    - PPTX = 幻灯片序号（1 起）；
    - DOCX = 逻辑页序号（段落累积约 1000 字符切一页，段落不拆开）——Word 无稳定分页。

    其他后缀抛 AppError(code=400)：入库只收这三种格式，宽进只会把乱码后缀喂给解析器，
    与其在解析器深处报一个看不懂的错，不如在入口用人话拒绝。
    """
    # 后缀大小写不敏感：Windows 用户常把 .PDF 扩展名改成大写
    suffix = path.suffix.lower()
    # 为什么延迟导入子解析器：子模块要 from app.ingest.parsers import ParsedPage，
    # 若在本模块顶层 import 子模块，「直接 import 子模块」的调用方式会触发
    # 半初始化包的循环导入；函数内导入把环彻底拆开。
    if suffix == ".pdf":
        from app.ingest.parsers.pdf_parser import parse_pdf

        return parse_pdf(path)
    if suffix == ".docx":
        from app.ingest.parsers.docx_parser import parse_docx

        return parse_docx(path)
    if suffix == ".pptx":
        from app.ingest.parsers.pptx_parser import parse_pptx

        return parse_pptx(path)
    raise AppError(message="不支持的文件类型（仅支持 PDF/Word/PPT）", code=400)


__all__ = ["ParsedPage", "parse_document"]
