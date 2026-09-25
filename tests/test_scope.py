# -*- coding: utf-8 -*-
"""
M4 质量优化单测：范围解析——页码区间/中文章节号/PDF 目录映射/降级提示。

为什么这些必须测死：范围是「总结越界」问题的根治手段——解析错一个字符，
用户拿到的就是错误范围的总结（比不过滤更糟：用户以为它懂了范围）。
"""
import pymupdf
import pytest

from app.agent.scope import _cn_to_int, _toc_range, parse_chapter, parse_page_range, resolve_scope
from app.db import crud


# ===== 页码区间 =====


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("总结第1-10页的内容", (1, 10)),
        ("第3页讲了什么", (3, 3)),
        ("看看第10到3页", (3, 10)),  # 笔误纠正为正序
        ("1~5页重点", (1, 5)),
        ("第一章讲了什么", None),  # 章节词不含「页」，不误判
        ("梯度下降是什么", None),
    ],
)
def test_parse_page_range(question, expected):
    assert parse_page_range(question) == expected


# ===== 章节号（含中文数字）=====


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("第一章讲了什么", (1, "章")),
        ("总结第十二章", (12, "章")),
        ("2.1 节的内容", None),  # 缺「第」字不认（避免误伤小节编号）
        ("第2节要点", (2, "节")),
        ("第十五章概述", (15, "章")),
        ("随便问问", None),
    ],
)
def test_parse_chapter(question, expected):
    assert parse_chapter(question) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("一", 1), ("七", 7), ("十", 10), ("十三", 13), ("二十", 20), ("三十五", 35), ("25", 25), ("abc", None)],
)
def test_cn_to_int(text, expected):
    assert _cn_to_int(text) == expected


# ===== TOC 区间计算 =====


def test_toc_range_covers_until_next_peer():
    """章节区间 = 本条 ~ 下一个同级/更高级目条前一页；末条到文档末页。"""
    toc = [
        [1, "第一章 绪论", 1],
        [1, "第二章 研究方法", 5],
        [2, "2.1 数据集", 6],
        [1, "第三章 实验", 8],
    ]
    assert _toc_range(toc, 1, "章", total_pages := 10) == (1, 4)
    assert _toc_range(toc, 2, "章", total_pages) == (5, 7)
    assert _toc_range(toc, 3, "章", total_pages) == (8, 10)  # 末条 → 文档末页
    assert _toc_range(toc, 9, "章", total_pages) is None  # 不存在的章节


# ===== 端到端 resolve（真 PDF + 真 DB 夹具）=====


def _make_pdf_with_toc(path, chapter_titles: list[str], pages: int = 6):
    """造一份带书签目录的 PDF（pymupdf set_toc，章节均分页码）。"""
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    step = pages // len(chapter_titles)
    toc = [[1, title, 1 + i * step] for i, title in enumerate(chapter_titles)]
    doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def kb_with_pdf(db_session, tmp_path):
    """用户 + 分组 + 一份带目录的 PDF 文档（resolve_scope 只读文件与文档行）。"""
    user = crud.create_user(db_session, username="scoper", password_hash="x")
    group = crud.create_kb_group(db_session, user_id=user.id, name="高数")
    pdf_path = tmp_path / "高等数学.pdf"
    _make_pdf_with_toc(pdf_path, ["第一章 函数", "第二章 极限", "第三章 导数"])
    doc = crud.create_document(
        db_session,
        user_id=user.id,
        group_id=group.id,
        file_name="高等数学.pdf",
        file_path=pdf_path,
        file_hash="0" * 64,
    )
    return {"db": db_session, "user": user, "group": group, "doc": doc}


def test_resolve_page_range_direct(kb_with_pdf):
    """显式页码优先：直接给出过滤条件，不碰文件。"""
    scope, note = resolve_scope(
        kb_with_pdf["db"],
        user_id=kb_with_pdf["user"].id,
        group_ids=[kb_with_pdf["group"].id],
        question="总结第1-10页",
    )
    assert scope == {"page_range": (1, 10)}
    assert note is None


def test_resolve_chapter_via_toc(kb_with_pdf):
    """「第一章」→ PDF 目录映射 → 页码区间 + 限定该文件。"""
    scope, note = resolve_scope(
        kb_with_pdf["db"],
        user_id=kb_with_pdf["user"].id,
        group_ids=[kb_with_pdf["group"].id],
        question="第一章讲了什么",
    )
    assert scope["page_range"] == (1, 2)  # 目录: 第1章@1, 第2章@3 → 区间 1~2
    assert scope["file_name"] == "高等数学.pdf"
    assert note is None


def test_resolve_chapter_missing_gives_note(kb_with_pdf):
    """章节在目录里找不到 → 不过滤 + 人话降级提示（静默降级=用户以为功能坏了）。"""
    scope, note = resolve_scope(
        kb_with_pdf["db"],
        user_id=kb_with_pdf["user"].id,
        group_ids=[kb_with_pdf["group"].id],
        question="第九十九章呢",
    )
    assert scope is None
    # 提示文案里的章节号是解析后的阿拉伯形式（第99章），并带页码问法引导
    assert note and "第99章" in note and "第X-Y页" in note


def test_resolve_no_scope_words_zero_overhead(db_session):
    """无范围词：(None, None)——绝大多数提问走这条路，零 IO 零提示。"""
    user = crud.create_user(db_session, username="plain", password_hash="x")
    scope, note = resolve_scope(db_session, user_id=user.id, group_ids=[], question="梯度下降是什么")
    assert scope is None and note is None
