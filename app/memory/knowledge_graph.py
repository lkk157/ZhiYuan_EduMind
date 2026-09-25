# -*- coding: utf-8 -*-
"""
知识点图谱（M5）：从课件元数据与向量索引**现算**的动态图——节点即文档、边即关联。

为什么是「派生现算」而不是「存边表」（产品经理明确要求动态，2026-09-25）：
节点的事实源是 documents 表（文件名自带章节编号），语义边的事实源是 Chroma 块向量——
两者随入库/删除实时变化，图谱每次打开现算**永远与真实课件同步**：
- 新上传 → 下次计算自动多出节点与边（编号解析即时、向量入库流水线本来就要写）；
- 删除 → 行没了节点自动消失，绝不会推荐已删章节；
- 同名重传 → 向量更新 → 语义边自动刷新。
若把边存成表，入库要写边、删档要清孤儿边、改名要改边——存下的图就是一份会腐烂的缓存。

三类成分、零标注、零 LLM：
1. 结构边（纯代码）：父节点（2.3.2→2.3「基础没牢」）、前序兄弟（2.3.2→2.3.1「顺序学习」）；
2. 语义边（纯本地）：取弱章节一个**存量块向量**做查询（连 embedding 都不打），
   在向量库里找内容最像的其他章节——真·语义关联，成本毫秒级；
3. 文件名无编号（讲义.docx/思维导图）→ 降级为文件节点，只有语义边。
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 文件名开头的章节编号：如 "2.3.2_单链表的插入删除" → code="2.3.2"。
# 兼容合并编号（3.4.1~3.4.4_xxx / 3.3.4+3.3.5_xxx）：只取第一段作为代表编号——
# 合并节本质是同一学习单元，图上占一个节点即可；
# 分隔符白名单必须含 ~ 与 +（实测漏了它们会把合并节降级成无编号文件节点）
_CODE_PATTERN = re.compile(r"^(?P<code>\d+(?:\.\d+)*)(?:[_\-\s~+]|$)")
# 编号后的分隔符与标题噪声
_TITLE_CLEAN = re.compile(r"^[_\-\s]+")


@dataclass
class ChapterNode:
    """图节点：一个课件文件（有编号=章节节点，无编号=文件节点）。"""

    file_name: str
    code: str  # 章节编号（"2.3.2"），无编号为空串
    title: str  # 展示标题（编号剥离后的文件名主体）
    depth: int = 0  # 编号层级（2.3.2 → 3），文件节点为 0

    @property
    def label(self) -> str:
        """展示文案：有编号带编号（用户按目录说话），无编号纯标题。"""
        return f"{self.code} {self.title}".strip() if self.code else self.title


@dataclass
class RelatedItem:
    """一条关联推荐：目标节点 + 关系类型 +（语义边才有）相似度。"""

    file_name: str
    label: str
    relation: str  # 前置基础 / 同章相邻 / 内容相近
    score: float | None = None


def parse_chapter(file_name: str) -> ChapterNode:
    """文件名 → 图节点（动态图的「长节点」入口，新上传的文档走这里）。

    为什么用文件名而不是解析 PDF 目录：文件名是入库契约的一部分（用户按
    教材目录命名），零 IO 即可得结构；PDF 目录有的文件有有的没有（还有 docx），
    文件名是唯一全覆盖的结构源。
    """
    stem = Path(file_name).stem
    m = _CODE_PATTERN.match(stem)
    if not m:
        # 无编号：文件节点（如「讲义」「数据结构思维导图【精简版】」）
        return ChapterNode(file_name=file_name, code="", title=stem, depth=0)
    code = m.group("code")
    # 先剥合并段残留（"3.4.1~3.4.4_xxx" 匹配到 3.4.1~ 后还剩 "3.4.4_xxx"，再剥一层编号段）
    rest = re.sub(r"^\d+(?:\.\d+)*[_\-\s~+]*", "", stem[m.end() :])
    title = _TITLE_CLEAN.sub("", rest) or stem
    return ChapterNode(file_name=file_name, code=code, title=title, depth=code.count(".") + 1)


def build_nodes(file_names: list[str]) -> list[ChapterNode]:
    """一批文件名 → 节点列表（同名去重；调用方来自 documents 表=动态来源）。"""
    seen: set[str] = set()
    nodes: list[ChapterNode] = []
    for name in file_names:
        if name in seen:
            continue
        seen.add(name)
        nodes.append(parse_chapter(name))
    return nodes


def structural_edges(nodes: list[ChapterNode]) -> list[tuple[ChapterNode, ChapterNode, str]]:
    """结构边：(子, 父/兄, 关系)。只连图上真实存在的节点（新文档自动纳入）。

    - 父边：2.3.2 → 2.3（编号去掉末段且存在的节点）——「往上找基础」；
    - 兄弟边：2.3.2 → 2.3.1（同父、编号末段-1 且存在）——「学习顺序上的前置」。
    两者都属于「复习推荐的结构依据」：错题卡壳处先补父、再补前兄。
    """
    by_code = {n.code: n for n in nodes if n.code}
    edges: list[tuple[ChapterNode, ChapterNode, str]] = []
    for node in nodes:
        if not node.code or "." not in node.code:
            continue  # 一级章（1.x 的「1」）没有父与兄可连（或父不存在）
        # 父：去末段
        parent_code = node.code.rsplit(".", 1)[0]
        parent = by_code.get(parent_code)
        if parent is not None:
            edges.append((node, parent, "前置基础"))
        # 前序兄弟：同父、末段-1（如 2.3.2 → 2.3.1）
        head, _, tail = node.code.rpartition(".")
        try:
            prev_num = int(tail) - 1
        except ValueError:
            continue
        if prev_num >= 1:
            sibling = by_code.get(f"{head}.{prev_num}")
            if sibling is not None:
                edges.append((node, sibling, "同章相邻"))
    return edges


def structural_related(nodes: list[ChapterNode], weak_file: str) -> list[RelatedItem]:
    """弱章节的结构推荐（父 + 前兄，按此顺序——先补基础再补顺序前置）。"""
    target = next((n for n in nodes if n.file_name == weak_file), None)
    if target is None:
        return []
    items: list[RelatedItem] = []
    for src, dst, relation in structural_edges(nodes):
        if src.file_name == weak_file:
            items.append(RelatedItem(file_name=dst.file_name, label=dst.label, relation=relation))
    # 去重保序（同一目标可能同时是父与兄——保留先出现的「前置基础」）
    seen: set[str] = set()
    unique: list[RelatedItem] = []
    for item in items:
        if item.file_name in seen:
            continue
        seen.add(item.file_name)
        unique.append(item)
    return unique


def semantic_related(
    user_id: int, weak_file: str, group_ids: list[int], *, exclude_files: set[str], top_k: int = 2
) -> list[RelatedItem]:
    """语义推荐：用弱章节的**存量块向量**做查询，找内容最像的其他章节。

    为什么用存量向量当查询（零 Ollama 调用）：同一块的向量已在库里，
    「与它最像的其他块」就是内容关联——再 embed 一遍纯属浪费；
    group_ids 由调用方（api 层，已开 db）传入——图谱模块不私自开数据库会话；
    查询失败/无锚一律返回 []（语义边是增强，挂了不影响结构边）。
    """
    from app.rag.vector_store import store_for as _store_for

    try:
        candidates: list[RelatedItem] = []
        for gid in group_ids:
            store = _store_for(user_id, gid)
            corpus = store.get_all()
            weak_ids = [h.id for h in corpus if str(h.metadata.get("file_name", "")) == weak_file]
            if not weak_ids:
                continue
            # 取第一块的存量向量做查询（同文件块方向一致，一块代表全节）
            vec = store.get_embeddings(weak_ids[:1]).get(weak_ids[0])
            if not vec:
                continue
            hits = store.query(vec, top_k * 6)  # 多取一些再跨组排序去重
            for hit in hits:
                file_name = str(hit.metadata.get("file_name", ""))
                if file_name == weak_file or file_name in exclude_files:
                    continue
                candidates.append(
                    RelatedItem(
                        file_name=file_name,
                        # 与结构边同款展示口径：有编号显示「编号 标题」（裸文件名带 .pdf 很碍眼）
                        label=parse_chapter(file_name).label,
                        relation="内容相近",
                        score=round(float(hit.score), 4),
                    )
                )
        # 跨组合并：按分数降序，文件去重，取 top_k
        candidates.sort(key=lambda x: -(x.score or 0.0))
        seen: set[str] = set()
        result: list[RelatedItem] = []
        for item in candidates:
            if item.file_name in seen:
                continue
            seen.add(item.file_name)
            result.append(item)
            if len(result) >= top_k:
                break
        return result
    except Exception:
        logger.exception("语义关联计算失败（降级：仅结构推荐）weak_file=%s", weak_file)
        return []


def related_for(
    user_id: int, weak_file: str, file_names: list[str], group_ids: list[int]
) -> list[RelatedItem]:
    """错题/薄弱章节 → 关联推荐总入口：结构边（父/兄）+ 语义边（内容相近）。

    动态性由 file_names 的来源保证：调用方每次从 documents 表现取，
    新上传的文档天然在列（产品经理需求：上传即入图）。
    """
    if not weak_file:
        return []
    nodes = build_nodes(file_names)
    if not any(n.file_name == weak_file for n in nodes):
        # 锚文件已删除（或从未入库）：图上没有弱节点，结构推荐无从谈起
        return []
    items = structural_related(nodes, weak_file)
    exclude = {i.file_name for i in items} | {weak_file}
    items += semantic_related(
        user_id, weak_file, group_ids, exclude_files=exclude
    )
    return items
