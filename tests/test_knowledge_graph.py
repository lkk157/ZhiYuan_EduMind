# -*- coding: utf-8 -*-
"""
M5 单测：知识点图谱——章节解析（含合并编号/无编号降级）、结构边、语义近邻、动态性。
"""
import pytest

from app.memory import knowledge_graph as kg

NODE_FILES = [
    "2.3_链表概述.pdf",
    "2.3.1_单链表的定义.pdf",
    "2.3.2_单链表的插入删除.pdf",
    "3.4.1~3.4.4_特殊矩阵的压缩存储.pdf",
    "3.3.4+3.3.5_队列的应用.pdf",
    "讲义.docx",
    "数据结构思维导图【精简版】.pdf",
]


def test_parse_chapter_variants():
    """编号/合并编号/无编号三类文件名的解析（合并段取第一段为节点编号并剥净标题）。"""
    normal = kg.parse_chapter("2.3.2_单链表的插入删除.pdf")
    assert (normal.code, normal.depth) == ("2.3.2", 3)
    assert normal.title == "单链表的插入删除"

    merged = kg.parse_chapter("3.4.1~3.4.4_特殊矩阵的压缩存储.pdf")
    assert merged.code == "3.4.1"
    assert merged.title == "特殊矩阵的压缩存储"  # 合并尾巴 3.4.4_ 必须剥掉

    plus = kg.parse_chapter("3.3.4+3.3.5_队列的应用.pdf")
    assert plus.code == "3.3.4"
    assert plus.title == "队列的应用"

    plain = kg.parse_chapter("数据结构思维导图【精简版】.pdf")
    assert plain.code == ""
    assert "思维导图" in plain.title  # 无编号 → 文件节点（降级但仍在图上）


def test_structural_edges_parent_and_sibling():
    """结构边：子→父（前置基础）、子→前序兄弟（同章相邻）；一号节点不越界。"""
    nodes = kg.build_nodes(NODE_FILES)
    edges = {(a.file_name, b.file_name, r) for a, b, r in kg.structural_edges(nodes)}
    assert ("2.3.2_单链表的插入删除.pdf", "2.3_链表概述.pdf", "前置基础") in edges
    assert ("2.3.2_单链表的插入删除.pdf", "2.3.1_单链表的定义.pdf", "同章相邻") in edges
    assert ("2.3.1_单链表的定义.pdf", "2.3_链表概述.pdf", "前置基础") in edges
    # 2.3 只有一段编号 → 无父无兄可连（它不会成为任何边的起点，不越出图外）
    sources_of_edges = {a.file_name for a, _, _ in kg.structural_edges(nodes)}
    assert "2.3_链表概述.pdf" not in sources_of_edges


def test_structural_related_order_and_unknown_file():
    """弱章节推荐 = 父在前兄在后；锚文件不在图上 → 空（动态删除后不越权推荐）。"""
    nodes = kg.build_nodes(NODE_FILES)
    items = kg.structural_related(nodes, "2.3.2_单链表的插入删除.pdf")
    assert [i.relation for i in items] == ["前置基础", "同章相邻"]
    assert items[0].label.startswith("2.3 ")

    assert kg.structural_related(nodes, "已删除的文件.pdf") == []


def test_related_for_dynamic_new_file_included():
    """动态性核心断言：新文件加进 file_names 列表（=新上传入库）→ 立即成为可推荐节点。"""
    base = kg.build_nodes(["2.3_链表概述.pdf", "2.3.1_单链表的定义.pdf"])
    assert kg.structural_related(base, "2.3.5_静态链表.pdf") == []  # 尚未入库

    grown = kg.build_nodes(
        ["2.3_链表概述.pdf", "2.3.1_单链表的定义.pdf", "2.3.5_静态链表.pdf"]
    )
    items = kg.structural_related(grown, "2.3.5_静态链表.pdf")
    # 新节点立刻有父边（2.3）；兄 2.3.4 不存在所以只有一条
    assert [(i.label, i.relation) for i in items] == [("2.3 链表概述", "前置基础")]


@pytest.mark.asyncio
async def test_related_for_semantic_excludes_self(monkeypatch, make_store):
    """语义近邻：存量向量查询（零 embedding）、排除自身与已有结构推荐目标。"""
    store = make_store()
    await store.upsert(
        ids=["1:0", "1:1", "1:2"],
        texts=[
            "单链表的插入删除操作详解",
            "单链表的插入删除补充说明",  # 与 1:0 同文件（弱锚）
            "双链表的前驱后继指针结构与插入",  # 语义近邻（异文件）
        ],
        metadatas=[
            {"file_name": "单链插入.pdf", "page_no": 1},
            {"file_name": "单链插入.pdf", "page_no": 2},
            {"file_name": "双链结构.pdf", "page_no": 1},
        ],
    )
    monkeypatch.setattr("app.rag.vector_store.store_for", lambda u, g, client=None: store)

    items = kg.semantic_related(1, "单链插入.pdf", [1], exclude_files={"单链插入.pdf"})
    assert items, "同文件外的语义近邻必须被找到"
    assert all(i.file_name != "单链插入.pdf" for i in items)  # 排除自身
    assert items[0].file_name == "双链结构.pdf"
    assert items[0].relation == "内容相近"
    # 零 embedding：本模块只用存量向量——embeddings.embed_texts 一被调用即失败
    monkeypatch.setattr(
        "app.rag.embeddings.embed_texts",
        lambda texts: (_ for _ in ()).throw(AssertionError("语义近邻严禁调用 embedding")),
    )
    again = kg.semantic_related(1, "单链插入.pdf", [1], exclude_files=set())
    assert again  # 再跑一遍依然成功（证明上一跑也没用 embedding）
