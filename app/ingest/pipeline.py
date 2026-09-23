# -*- coding: utf-8 -*-
"""
入库流水线（写路径）：文件 -> 逐页解析 -> 页感知切块 -> 指纹 diff -> 增量向量化 -> 落库落索引。

为什么流水线独立成层（api 层不许自己编排这套步骤）：
1. 同名重传的「chunk 级增量」是本系统最绕的状态机（DB 文档行 / 块指纹 / Chroma 三方
   必须一致），收口在一个函数里才可能测全；
2. 失败留痕（status=failed）与幂等短路（skipped_identical）是写路径的两条底线，
   散落在接口里迟早漏一条；
3. 向量化只发生在 added+changed 上（store.upsert 内部经 app.rag.embeddings -> gateway），
   重传成本与改动量成正比——gateway 的显存串行化/批量限流（CLAUDE.md §3）自动生效，
   本层不直接打 Ollama。
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.crud import (
    create_document,
    get_document_by_name,
    list_chunk_fingerprints,
    replace_chunk_fingerprints,
    update_document_meta,
)
from app.db.models import Document
from app.ingest.chunker import chunk_pages
from app.ingest.fingerprint import chunk_sha256, diff_chunks, file_sha256
from app.ingest.parsers import parse_document
from app.rag import vector_store
from app.rag.vector_store import ChromaStore, store_for

logger = logging.getLogger(__name__)

# 导入时快照原始实现：_open_store 靠它判断 store_for 是否被单测打过补丁（见下方双缝说明）
_store_for_original = store_for


@dataclass
class IngestResult:
    """一次入库的结果计数（HTTP 出口直接透传给前端展示）。

    为什么 skipped_identical 是独立字段而不是「所有计数为 0」的歧义表达：
    「内容没变被跳过」和「空文档入库」在计数上都是 0，但用户提示语义完全不同，
    必须显式区分（前者提示「已跳过」，后者提示「未解析到文本」）。
    """

    doc_id: int
    skipped_identical: bool
    added: int
    changed: int
    removed: int
    unchanged: int
    page_count: int
    chunk_count: int
    empty_pages: list[int]


def _open_store(user_id: int, group_id: int) -> ChromaStore:
    """打开「用户-分组」向量库（双缝兼容单测注入，运行时解析 store_for）。

    为什么要做双缝解析：ingest_file 的层间契约签名没有 stores/client 注入参数，
    卽测必须靠 monkeypatch 塞 EphemeralClient 版假向量库，而补丁落点习惯有两种——
    patch 定义处（app.rag.vector_store.store_for，与 embeddings.embed_texts 的约定一致）
    或 patch 调用方（app.ingest.pipeline.store_for）。
    这里优先用本模块命名空间的 store_for（调用方被打补丁即生效），
    否则回退 vector_store.store_for（定义处被打补丁也能生效）。
    两种姿势都兼容，避免测试补丁静默失效、真 PersistentClient 落盘。
    """
    fn = store_for if store_for is not _store_for_original else vector_store.store_for
    return fn(user_id, group_id)


def _mark_failed(
    db: Session,
    *,
    user_id: int,
    group_id: int,
    file_name: str,
    file_path: Path,
    file_hash: str,
    doc: Document | None,
) -> None:
    """入库失败后把文档标成 failed；新建文档尚无行时补一行留痕。

    为什么新建失败也要留行：不留痕的话用户只看到「上传失败」的瞬时报错，
    刷新页面后无从得知哪份资料没进去；留一行 status=failed，
    前端文档列表能稳定展示「导入失败，请重新上传」。
    """
    if doc is None:
        create_document(
            db,
            user_id=user_id,
            group_id=group_id,
            file_name=file_name,
            file_path=file_path,
            file_hash=file_hash,
            status="failed",
        )
    else:
        # 已有行只改 status：失败时向量库/指纹可能停留在旧版本上，
        # 保留旧的 page_count 等元数据比清零更贴近索引现状
        update_document_meta(db, doc, status="failed")


async def ingest_file(
    db: Session,
    *,
    user_id: int,
    group_id: int,
    file_name: str,
    file_path: Path,
) -> IngestResult:
    """把一个上传文件增量入库（同名重传 = 同一文档的 chunk 级版本更新）。

    流程（层间契约规定的次序，不可换序）：
    a) file_sha256 幂等短路——同名 + 同哈希 + status=ready 直接返回 skipped_identical；
    b) parse_document 逐页解析 -> chunk_pages(settings.chunk_size, settings.chunk_overlap)；
    c) 读旧块指纹 {chunk_index: chunk_hash} -> diff_chunks 四分类；
    d) 只对 added+changed 向量化并 upsert（向量 id="{doc_id}:{chunk_index}"）；
    e) removed 的 chunk_index 逐个 delete 向量；
    f) replace_chunk_fingerprints 整组重写为 diff 后最终集合（含 unchanged）；
    g) create/update Document（file_hash/页数/块数/empty_pages/status=ready）；
    h) 任何解析/向量化异常：Document.status 置 failed（新建也留痕）后原样上抛。

    为什么 (user_id, group_id, file_name) 唯一约束下还要自己查重：
    唯一约束只是并发兜底；单测用 SQLite、生产用 MySQL，业务判断必须自己做，
    不能依赖某个数据库的约束报错时机。
    """
    # a) 幂等短路：整文件哈希是「内容完全没变」的最便宜判据（先哈希后解析，省一大轮）
    file_hash = file_sha256(file_path)
    existing = get_document_by_name(
        db, user_id=user_id, group_id=group_id, file_name=file_name
    )
    if existing is not None and existing.file_hash == file_hash and existing.status == "ready":
        # 契约口径：skipped_identical=True 时「其余计数 0」——本轮没做任何入库动作，
        # 计数一律 0（含 page_count/chunk_count），doc_id 供前端定位文档行
        return IngestResult(
            doc_id=existing.id,
            skipped_identical=True,
            added=0,
            changed=0,
            removed=0,
            unchanged=0,
            page_count=0,
            chunk_count=0,
            empty_pages=[],
        )

    doc: Document | None = existing
    try:
        # b) 解析 + 切块（块绝不跨页；空文本页不产块，占位文本绝不进向量库）
        pages, empty_pages = parse_document(file_path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

        if doc is None:
            # 新文档必须先落行：向量 id="{document_id}:{chunk_index}" 依赖 document_id，
            # 先有行才有稳定 id（同名重传靠这个 id 复用/覆盖旧块）
            doc = create_document(
                db,
                user_id=user_id,
                group_id=group_id,
                file_name=file_name,
                file_path=file_path,
                file_hash=file_hash,
                page_count=len(pages),
                chunk_count=len(chunks),
                empty_pages=empty_pages,
                status="ready",
            )
            old_fingerprints: dict[int, str] = {}
        else:
            # 同名重传：旧指纹是增量基线，按 chunk_index 组装 diff 输入
            old_fingerprints = {
                fp.chunk_index: fp.chunk_hash
                for fp in list_chunk_fingerprints(db, document_id=doc.id)
            }

        # c) 严格四分类（按 chunk_index 对齐比较 hash）
        diff = diff_chunks(old_fingerprints, chunks)

        # d) 只对 added+changed 向量化并 upsert：
        #    ChromaStore.upsert 内部经 embeddings.embed_texts 向量化（gateway 限流），
        #    unchanged 块不重算不重写——重传成本只与改动量成正比
        store = _open_store(user_id, group_id)
        dirty = diff.added + diff.changed
        if dirty:
            await store.upsert(
                ids=[f"{doc.id}:{chunk.chunk_index}" for chunk in dirty],
                texts=[chunk.text for chunk in dirty],
                # metadata 键名是层间契约（retriever 溯源按这些键取值），禁止增删改名
                metadatas=[
                    {
                        "file_name": file_name,
                        "page_no": chunk.page_no,
                        "chunk_index": chunk.chunk_index,
                        "doc_id": doc.id,
                        "group_id": group_id,
                        "user_id": user_id,
                    }
                    for chunk in dirty
                ],
            )

        # e) 被删块的向量同步删除：留着孤儿块会在检索里引用出「已不存在的内容」
        if diff.removed:
            store.delete([f"{doc.id}:{index}" for index in diff.removed])

        # f) 指纹整组替换为「diff 后最终集合」（added/changed/unchanged 全量重写），
        #    保证指纹表 == 当前块集合，removed 的陈旧指纹一并清掉
        replace_chunk_fingerprints(
            db,
            document_id=doc.id,
            fingerprints=[(chunk.chunk_index, chunk_sha256(chunk.text)) for chunk in chunks],
        )

        # g) 文档元数据收尾：status=ready 才允许被检索命中
        update_document_meta(
            db,
            doc,
            file_path=file_path,
            file_hash=file_hash,
            page_count=len(pages),
            chunk_count=len(chunks),
            empty_pages=empty_pages,
            status="ready",
        )
        return IngestResult(
            doc_id=doc.id,
            skipped_identical=False,
            added=len(diff.added),
            changed=len(diff.changed),
            removed=len(diff.removed),
            unchanged=len(diff.unchanged),
            page_count=len(pages),
            chunk_count=len(chunks),
            empty_pages=empty_pages,
        )
    except Exception:
        # h) 失败留痕后原样上抛：解析失败/向量化失败（Ollama 挂了）都必须让前端
        #    看到 failed 而不是半截 ready，api 层再把异常翻成人话
        try:
            _mark_failed(
                db,
                user_id=user_id,
                group_id=group_id,
                file_name=file_name,
                file_path=file_path,
                file_hash=file_hash,
                doc=doc,
            )
        except Exception:
            # 留痕动作自身失败绝不能掩盖原始异常——否则排查看不到真正的报错
            logger.exception("入库失败后写 status=failed 时再次出错")
        raise


__all__ = ["IngestResult", "ingest_file"]
