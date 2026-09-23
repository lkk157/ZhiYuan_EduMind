# -*- coding: utf-8 -*-
"""
数据读写封装（CRUD）：api 层不直接写 SQL，统一走这里。

为什么要有这一层：
1. SQL 散落在接口里会和参数校验/错误处理缠成一团，难测难改；
2. 唯一约束冲突这类数据库细节在这里翻译成业务异常（ConflictError），
   接口层只管用人话回给前端；
3. 非本人资源的读写一律「带 user_id 过滤」——查不到就返回 None/False，
   由 api 层转 NotFoundError，防止通过 id 猜测探测他人数据（越权防线前移到数据层）。
"""
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError
from app.db.models import ChunkFingerprint, Document, KbGroup, User, join_empty_pages


def _translate_unique_conflict(e: IntegrityError, message: str) -> None:
    """把「唯一约束冲突」翻译成 ConflictError，其他完整性错误如实上抛。

    为什么只精确匹配 unique/duplicate 关键词（M1 单测实测踩过的坑）：
    缺字段、外键失败等完整性错误必须原样上抛，否则真 bug 会被误报成业务冲突。
    SQLite 报 "UNIQUE constraint failed"，MySQL 报 "Duplicate entry"。
    """
    detail = str(getattr(e, "orig", e)).lower()
    if "unique" in detail or "duplicate" in detail:
        raise ConflictError(message) from e
    raise


# ===== 用户（M1 既有能力，签名与行为保持不变）=====


def create_user(db: Session, username: str, password_hash: str) -> User:
    """创建用户。用户名撞车时抛 ConflictError（而不是泄漏数据库异常细节）。"""
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()  # 冲突后必须回滚，否则会话进入坏状态后续操作全报错
        _translate_unique_conflict(e, "用户名已被注册")
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    """按用户名查用户（登录用）。"""
    return db.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """按 id 查用户（JWT 解析后回查，确认账号仍存在且归属正确）。"""
    return db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()


# ===== 知识库分组 =====


def create_kb_group(db: Session, *, user_id: int, name: str) -> KbGroup:
    """创建知识库分组。同一用户下分组名撞车抛 ConflictError「分组名已存在」。

    唯一冲突人话文案固定为「分组名已存在」：前端对 409 可直接透传给用户。
    """
    group = KbGroup(user_id=user_id, name=name)
    db.add(group)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()  # 冲突后回滚，避免会话进入坏状态
        _translate_unique_conflict(e, "分组名已存在")
    db.refresh(group)
    return group


def list_kb_groups(db: Session, *, user_id: int) -> list[KbGroup]:
    """列出某用户的全部分组（按 id 升序，保证前端列表顺序稳定可预期）。"""
    rows = db.execute(
        select(KbGroup).where(KbGroup.user_id == user_id).order_by(KbGroup.id)
    ).scalars()
    return list(rows)


def get_kb_group(db: Session, *, user_id: int, group_id: int) -> KbGroup | None:
    """按 id 查分组，强制带 user_id 过滤。

    为什么必须双条件：别人的分组一律查不到（返回 None，api 层转 NotFoundError 防探测），
    不能只按 group_id 查——否则等于公开全站分组的存在性。
    """
    return db.execute(
        select(KbGroup).where(KbGroup.id == group_id, KbGroup.user_id == user_id)
    ).scalar_one_or_none()


def delete_kb_group(db: Session, *, user_id: int, group_id: int) -> bool:
    """删除分组，连带删其下全部文档与块指纹行；非本人/不存在返回 False。

    为什么在这里手工级联（而不是只依赖外键 ON DELETE CASCADE）：
    单测用 SQLite 时外键约束默认不生效，手工按「指纹 → 文档 → 分组」顺序删干净，
    保证 MySQL/SQLite 两种存储下都不留孤儿行。
    上传文件与 Chroma collection 不属于 DB 事务，由 api 层另行清理。
    """
    group = get_kb_group(db, user_id=user_id, group_id=group_id)
    if group is None:
        return False
    doc_ids = select(Document.id).where(Document.group_id == group.id)
    # 先删指纹再删文档再删分组：子表清空后父行删除，顺序不可反
    db.execute(delete(ChunkFingerprint).where(ChunkFingerprint.document_id.in_(doc_ids)))
    db.execute(delete(Document).where(Document.group_id == group.id))
    db.delete(group)
    db.commit()
    return True


# ===== 文档 =====


def get_document_by_name(
    db: Session, *, user_id: int, group_id: int, file_name: str
) -> Document | None:
    """按 (user_id, group_id, file_name) 查文档——同名重传入口先调它判断走增量还是新建。"""
    return db.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.group_id == group_id,
            Document.file_name == file_name,
        )
    ).scalar_one_or_none()


def create_document(
    db: Session,
    *,
    user_id: int,
    group_id: int,
    file_name: str,
    file_path: str | Path,
    file_hash: str,
    page_count: int = 0,
    chunk_count: int = 0,
    empty_pages: list[int] | None = None,
    status: str = "ready",
) -> Document:
    """新建文档行。同名撞车抛 ConflictError（正常流程应先 get_document_by_name 走增量更新）。

    file_path 接受 pathlib.Path 并转字符串存库：Windows 路径统一走 pathlib（CLAUDE.md §5），
    存储层仍保持字符串以保证两库同构。
    empty_pages 收 list[int]，在模型层编码成逗号拼接字符串（见 join_empty_pages 注释）。
    """
    doc = Document(
        user_id=user_id,
        group_id=group_id,
        file_name=file_name,
        file_path=str(file_path),
        file_hash=file_hash,
        page_count=page_count,
        chunk_count=chunk_count,
        empty_pages=join_empty_pages(empty_pages or []),
        status=status,
    )
    db.add(doc)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        _translate_unique_conflict(e, "同名文档已存在")
    db.refresh(doc)
    return doc


def list_documents(db: Session, *, user_id: int, group_id: int) -> list[Document]:
    """列出分组下全部文档（按 id 升序=上传先后，前端列表稳定）。"""
    rows = db.execute(
        select(Document)
        .where(Document.user_id == user_id, Document.group_id == group_id)
        .order_by(Document.id)
    ).scalars()
    return list(rows)


def get_document(db: Session, *, user_id: int, doc_id: int) -> Document | None:
    """按 id 查文档，强制带 user_id 过滤（越权防线同 get_kb_group）。"""
    return db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    ).scalar_one_or_none()


def update_document_meta(
    db: Session,
    document: Document,
    *,
    file_path: str | Path | None = None,
    file_hash: str | None = None,
    page_count: int | None = None,
    chunk_count: int | None = None,
    empty_pages: list[int] | None = None,
    status: str | None = None,
) -> Document:
    """增量更新文档元数据（版本更新/入库收尾/失败落状态）。

    为什么传 None 表示「不改」：入库流水线各阶段掌握的字段不同
    （解析完才知道页数、切块完才知道块数、失败时只知道 status），
    部分更新免去调用方拼全量参数；updated_at 由模型 onupdate 自动刷新。
    """
    if file_path is not None:
        document.file_path = str(file_path)
    if file_hash is not None:
        document.file_hash = file_hash
    if page_count is not None:
        document.page_count = page_count
    if chunk_count is not None:
        document.chunk_count = chunk_count
    if empty_pages is not None:
        document.empty_pages = join_empty_pages(empty_pages)
    if status is not None:
        document.status = status
    db.commit()
    db.refresh(document)
    return document


def delete_document(db: Session, *, user_id: int, doc_id: int) -> bool:
    """删除文档并连带删其块指纹行；非本人/不存在返回 False。

    指纹必须与文档同生共死：留着孤儿指纹会让下次同名重建文档时 diff 误判增量。
    对应向量的删除属于 Chroma 层，由 api 层按 document_id 前缀另行清理。
    """
    doc = get_document(db, user_id=user_id, doc_id=doc_id)
    if doc is None:
        return False
    db.execute(delete(ChunkFingerprint).where(ChunkFingerprint.document_id == doc.id))
    db.delete(doc)
    db.commit()
    return True


# ===== 块指纹（chunk 级增量的比对基线）=====


def list_chunk_fingerprints(db: Session, *, document_id: int) -> list[ChunkFingerprint]:
    """列出某文档当前全部块指纹（按 chunk_index 升序）。

    调用方用 {fp.chunk_index: fp.chunk_hash} 组装成 diff_chunks 需要的 old 映射。
    """
    rows = db.execute(
        select(ChunkFingerprint)
        .where(ChunkFingerprint.document_id == document_id)
        .order_by(ChunkFingerprint.chunk_index)
    ).scalars()
    return list(rows)


def replace_chunk_fingerprints(
    db: Session, *, document_id: int, fingerprints: list[tuple[int, str]]
) -> None:
    """整组替换某文档的块指纹为「diff 后的最终有效集合」。

    为什么是先清后写而不是逐条 upsert：diff 的 removed 块必须从指纹表消失，
    整组替换保证指纹表 == 当前块集合，不留陈旧指纹导致下次重传误判（把已删块当成 unchanged）。
    fingerprints 为 (chunk_index, chunk_hash) 列表；空列表=文档被清空，全部指纹删除。
    """
    db.execute(
        delete(ChunkFingerprint).where(ChunkFingerprint.document_id == document_id)
    )
    for chunk_index, chunk_hash in fingerprints:
        db.add(
            ChunkFingerprint(
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_hash=chunk_hash,
            )
        )
    db.commit()


def delete_chunk_fingerprints(db: Session, *, document_id: int) -> int:
    """单独清空某文档的块指纹（入库失败回滚增量基线时用），返回删除行数。"""
    result = db.execute(
        delete(ChunkFingerprint).where(ChunkFingerprint.document_id == document_id)
    )
    db.commit()
    return result.rowcount or 0
