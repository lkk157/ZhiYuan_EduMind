# -*- coding: utf-8 -*-
"""
指纹层单测（M2）：两级 sha256 + chunk 级四分类 diff——增量入库的判据。

为什么这块是生产思维的落点：课件「改几页再传」是常态，
四分类判错任何一类，要么浪费一堆向量重算（判不准 changed），
要么检索引用出已删内容（漏判 removed）——必须测死。
"""
from app.ingest.chunker import Chunk
from app.ingest.fingerprint import chunk_sha256, diff_chunks, file_sha256


def test_chunk_sha256_stable_and_distinct():
    """同文本必同哈希，不同文本必不同哈希（原文进原文出，不归一化）。"""
    assert chunk_sha256("知源") == chunk_sha256("知源")
    assert chunk_sha256("知源") != chunk_sha256("知源 ")


def test_file_sha256_stable_and_distinct(tmp_path):
    """文件级指纹：同内容稳定（幂等短路的依据），不同内容不同。"""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello zhiyuan")
    b.write_bytes(b"hello zhiyuan")
    assert file_sha256(a) == file_sha256(b)
    b.write_bytes(b"hello zhiyuan!")
    assert file_sha256(a) != file_sha256(b)


def test_diff_four_way_classification():
    """四分类主场景：改 0、同 1、旧 2 消失、新 3 出现——不重不漏。"""
    t0_old, t0_new, t1, t2, t3 = "alpha", "alpha-changed", "beta", "gamma", "delta"
    old = {0: chunk_sha256(t0_old), 1: chunk_sha256(t1), 2: chunk_sha256(t2)}
    new = [Chunk(0, 1, t0_new), Chunk(1, 1, t1), Chunk(3, 1, t3)]

    diff = diff_chunks(old, new)
    assert [c.text for c in diff.changed] == [t0_new]
    assert [c.chunk_index for c in diff.changed] == [0]
    assert diff.unchanged == [1]
    assert [c.chunk_index for c in diff.added] == [3]
    assert diff.removed == [2]


def test_diff_empty_old_is_all_added():
    """全新文档：旧指纹为空 → 全部 added（首轮入库的主路径）。"""
    chunks = [Chunk(0, 1, "x"), Chunk(1, 1, "y")]
    diff = diff_chunks({}, chunks)
    assert [c.chunk_index for c in diff.added] == [0, 1]
    assert diff.changed == []
    assert diff.removed == []
    assert diff.unchanged == []


def test_diff_identical_is_all_unchanged():
    """逐字节相同的重传（文件级短路失效时的兜底）：全 unchanged，零向量重算。"""
    chunks = [Chunk(0, 1, "x"), Chunk(1, 1, "y")]
    old = {0: chunk_sha256("x"), 1: chunk_sha256("y")}
    diff = diff_chunks(old, chunks)
    assert diff.unchanged == [0, 1]
    assert diff.added == []
    assert diff.changed == []
    assert diff.removed == []


def test_diff_removed_sorted():
    """removed 排序输出：结果稳定可断言（不依赖字典插入顺序）。"""
    old = {5: chunk_sha256("a"), 2: chunk_sha256("b"), 9: chunk_sha256("c")}
    diff = diff_chunks(old, [Chunk(0, 1, "new")])
    assert diff.removed == [2, 5, 9]
    assert [c.chunk_index for c in diff.added] == [0]
