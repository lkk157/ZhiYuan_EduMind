# -*- coding: utf-8 -*-
"""
指纹层：文件 / 块哈希与 chunk 级四分类 diff，是增量入库的判据。

为什么用哈希而不是直接比文本：
哈希定长 64 字符可入库可比较，比对代价与块数成正比；直接存/比全文既费空间又慢。
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunker import Chunk


def file_sha256(path: Path) -> str:
    """整文件字节流的 sha256 十六进制（同名重传的幂等短路依据）。

    为什么分块读（1MB）而不是 read() 整读：大课件一次整读会把整个文件塞进内存，
    答辩机只有 16G 内存（CLAUDE.md §3 硬件红线），流式哈希内存占用恒定。
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)  # 1MB 一块
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def chunk_sha256(text: str) -> str:
    """文本原文（utf-8 编码）的 sha256 十六进制。

    为什么不归一化（去空白 / 统一大小写）后再哈希——保持简单确定：
    原文进原文出，同文本必同哈希；归一化最多省下几条重复向量，
    却会引入「归一后相同但原文不同」的边界争议，得不偿失。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ChunkDiff:
    """四分类增量结果：added / changed 带块本体（要重算向量），removed / unchanged 只带索引。

    为什么 removed / unchanged 是 list[int] 而不是 list[Chunk]：
    它们不需要块文本——removed 只要索引去删向量，unchanged 只要计数上报，
    携带块本体纯属浪费内存（大文档块数上千）。
    """

    added: list[Chunk]
    changed: list[Chunk]
    removed: list[int]
    unchanged: list[int]


def diff_chunks(old: dict[int, str], new: list[Chunk]) -> ChunkDiff:
    """按 chunk_index 对齐比较哈希，严格四分类（added / changed / removed / unchanged）。

    分类规则（严格，不重不漏）：
    - 新有旧无 -> added（带块本体，待向量化）；
    - 新有旧有且哈希不同 -> changed（带块本体，待重算向量并覆盖）；
    - 新有旧有且哈希相同 -> unchanged（只记索引，向量与 metadata 原样保留）；
    - 旧有新无 -> removed（只记索引，待删向量）。

    为什么按 chunk_index 对齐而不是按内容找「移动过的块」：
    chunk_index 是向量稳定命名（"{document_id}:{chunk_index}"）的一部分，
    同索引同哈希 = 该位置块没动，零成本保留；中间插入内容导致后续块整体移位时，
    会保守地把移位块判成 changed（多算几条向量不影响正确性），
    而「跨位置找移动块」的复杂匹配不值得——教案重传以局部改动为主。

    参数 old 为旧指纹映射 {chunk_index: chunk_hash}，new 为新切出的块列表。
    """
    added: list[Chunk] = []
    changed: list[Chunk] = []
    unchanged: list[int] = []
    seen: set[int] = set()
    for chunk in new:
        seen.add(chunk.chunk_index)
        new_hash = chunk_sha256(chunk.text)
        if chunk.chunk_index not in old:
            added.append(chunk)  # 旧文档没有这个位置：全新块
        elif old[chunk.chunk_index] != new_hash:
            changed.append(chunk)  # 同位置不同哈希：内容变了，必须重算向量
        else:
            unchanged.append(chunk.chunk_index)  # 同位置同哈希：原样保留
    # removed 排序输出：结果稳定可断言（不依赖 old 字典的插入顺序）
    removed = sorted(index for index in old if index not in seen)
    return ChunkDiff(added=added, changed=changed, removed=removed, unchanged=unchanged)
