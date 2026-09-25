# -*- coding: utf-8 -*-
"""
范围检索（质量优化第 2 层）：把「第1-10页」「第一章」解析成检索过滤条件。

为什么需要它：语义检索是「平铺 top_k」——问「总结第一章」时召回的块可能散布全书，
模型拿到哪 5 块就总结哪 5 块，用户要的「范围」完全不被尊重（2026-09-25 实测反馈）。
页码范围是显式、零歧义的过滤条件；章节则靠 PDF 目录（书签）在提问时按需映射成页码。

为什么章节映射在「提问时」而不是「入库时」：
1. 入库时解析 = 给 documents 表加目录列 → 已有表加列要 SQLite/MySQL 双份迁移（禁）；
2. 提问时按需读文件目录 = 零表结构变更，且只有含章节词的提问才付这次 IO 成本；
3. PDF 目录只在文件里（pymupdf get_toc），事实源就是文件本身，不存在两处同步问题。

降级口径：文件无目录信息 / 章节号对不上 → 不过滤（行为退回优化前），但必须给
scope_note 人话提示（「请改用第X-Y页提问」）——静默降级会让用户以为功能坏了。
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 「第1-10页」「1到10页」「第1~10页」→ 页码区间；「第3页」→ 单页区间
# 顺序敏感：区间模式必须先于单页模式尝试，否则「第1-10页」会被单页模式截成「第1页」
_PAGE_RANGE_PATTERNS = (
    re.compile(r"第?\s*(\d{1,4})\s*[-–~～到至]\s*(\d{1,4})\s*页"),
    re.compile(r"第\s*(\d{1,4})\s*页"),
)
# 「第一章」「第2节」——数字支持阿拉伯与中文（两/二都认）
_CHAPTER_PATTERN = re.compile(r"第\s*([0-9一二两三四五六七八九十]+)\s*([章节])")

# 中文数字 → int（支持 到 九十九：十、十一、二十、三十五……）
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(text: str) -> int | None:
    """中文/阿拉伯数字统一转 int；无法识别返回 None（调用方按「解析失败」降级）。"""
    if text.isdigit():
        return int(text)
    if text in _CN_DIGITS:
        return _CN_DIGITS[text]
    if text == "十":
        return 10
    if text.startswith("十"):  # 十一 ~ 十九
        rest = text[1:]
        return 10 + _CN_DIGITS.get(rest, 0) if rest in _CN_DIGITS else None
    if "十" in text:  # 二十 / 三十五
        head, _, tail = text.partition("十")
        tens = _CN_DIGITS.get(head, 0) if head else 1
        # 「十五」head=「五」? 不会——partition 前半是「五」时原文是「五十」的镜像，
        # 中文规则里十位在前：「三十五」→ head=三 tail=五；head 只可能是 1-9
        if head not in _CN_DIGITS:
            return None
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    return None


def parse_page_range(question: str) -> tuple[int, int] | None:
    """从提问里解析页码范围；没有页码词返回 None（不构成过滤条件）。"""
    for pattern in _PAGE_RANGE_PATTERNS:
        m = pattern.search(question)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:  # 区间
            a, b = int(m.group(1)), int(m.group(2))
        else:  # 单页
            a = b = int(m.group(1))
        if a > b:  # 「第10-3页」是笔误，纠正而非拒绝
            a, b = b, a
        return (a, b)
    return None


def parse_chapter(question: str) -> tuple[int, str] | None:
    """解析「第一章 / 第2节」→ (编号, 标记)；没有章节词返回 None。"""
    m = _CHAPTER_PATTERN.search(question)
    if not m:
        return None
    num = _cn_to_int(m.group(1))
    if num is None:
        return None
    return (num, m.group(2))


def _toc_range(toc: list, num: int, marker: str, total_pages: int) -> tuple[int, int] | None:
    """在 PDF 目录里找「第{num}{marker}」的页码区间。

    区间 = 本条目页码 ~ 下一个同级或更高级目条的前一页；最后一条到文档末页。
    只按「编号+标记」匹配（标题文字可含任意章节名）——「第一章 绪论」「1 绪论」都认。
    """
    # toc 元素: [level, title, page]
    entries = [(lvl, str(title), int(page)) for lvl, title, page in toc if int(page) > 0]
    target = None
    for idx, (lvl, title, page) in enumerate(entries):
        m = _CHAPTER_PATTERN.search(title)
        if not m:
            continue
        t_num = _cn_to_int(m.group(1))
        if t_num == num and m.group(2) == marker:
            target = (idx, lvl, page)
            break
    if target is None:
        return None
    idx, lvl, start = target
    end = total_pages
    for nxt_lvl, _, nxt_page in entries[idx + 1 :]:
        if nxt_lvl <= lvl:  # 同级或更高级的下一条 = 本章节结束边界
            end = nxt_page - 1
            break
    if end < start:
        end = start
    return (start, end)


def resolve_scope(db, *, user_id: int, group_ids: list[int], question: str) -> tuple[dict | None, str | None]:
    """把提问中的范围词解析成检索过滤条件。

    返回 (scope, scope_note)：
    - scope=None, note=None：提问无范围词（绝大多数情况，零开销直接返回）；
    - scope={"page_range": (a,b), "file_name": 可选}，note=None：解析成功；
    - scope=None, note=人话提示：用户要了范围但解析不了（无目录/章节对不上）——
      降级为不过滤，但必须把原因告诉用户（见模块 docstring）。

    显式页码优先于章节（「第1-10页的第二章」按页码走，页码更硬）。
    """
    page_range = parse_page_range(question)
    if page_range is not None:
        return {"page_range": page_range}, None

    chapter = parse_chapter(question)
    if chapter is None:
        return None, None
    num, marker = chapter

    # 遍历所选分组里的 PDF（只有 PDF 有书签目录；Word/PPT 的目录结构无稳定读取口径）
    from app.db import crud  # 函数内导入：避免 scope 被 crud 单测引用时产生无关依赖边

    for gid in group_ids:
        for doc in crud.list_documents(db, user_id=user_id, group_id=gid):
            path = Path(doc.file_path)
            if path.suffix.lower() != ".pdf" or not path.is_file():
                continue
            try:
                import pymupdf

                with pymupdf.open(str(path)) as pdf:
                    toc = pdf.get_toc()
                    total = pdf.page_count
            except Exception:
                logger.info("读取 PDF 目录失败，跳过: %s", path.name)
                continue
            if not toc:
                continue  # 无目录的 PDF 继续看下一份
            rng = _toc_range(toc, num, marker, total)
            if rng is not None:
                # 限定到该文件：章节属于某本书，跨文件混合会让页码区间失去意义
                return {"page_range": rng, "file_name": doc.file_name}, None

    # 走到这里 = 用户要了章节但映射失败：不过滤 + 人话提示
    return None, (
        f"课件中未找到「第{num}{marker}」的目录信息，本次已按全部资料检索；"
        "若要精确限定范围，请改用「第X-Y页」的问法。"
    )
