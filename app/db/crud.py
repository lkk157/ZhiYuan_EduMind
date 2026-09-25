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

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError
from app.db.models import (
    ChunkFingerprint,
    Conversation,
    Document,
    KbGroup,
    MemoryFact,
    Message,
    QuizRecord,
    User,
    join_empty_pages,
)


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


# ===== 会话历史（RAG 完善阶段：问答记录落 MySQL 事实源）=====


def create_conversation(db: Session, *, user_id: int, title: str) -> Conversation:
    """新建空会话。标题由调用方给（前端截首问前 30 字），空标题由接口层兜底成「新对话」。

    为什么这里不做重名校验：会话允许重名（与分组唯一约束语义不同，理由见模型注释）。
    """
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db: Session, *, user_id: int) -> list[tuple[Conversation, int]]:
    """列出某用户全部会话，附各会话消息数，按最近活跃倒序。

    排序键三级：updated_at DESC → MAX(message.id) DESC → conversation.id DESC。

    为什么一次 JOIN 查出消息数而不是逐会话 len(list_messages)：
    会话列表页要 N 次查询才能渲染完（N+1 问题）；这里用聚合一次查完——
    与 kb.list_groups 的逐组 N+1 不同，那是 M2 已验收的既有代码不动，
    新代码按正确姿势写，不复制旧毛病。

    为什么第二键是 MAX(Message.id)（单测实测踩坑）：updated_at 精度在
    SQLite/MySQL 默认都是秒级——同一秒里「刚问答过的旧会话」和「刚建的新会话」
    时间戳并列，若只用 conversation.id 兜底，刚问答的会话反而被压在下面。
    消息 id 自增单调、粒度远细于秒，coalesce 成 0 后「有消息的」恒排「空会话」前面，
    同秒并列也能严格分出活跃先后；第三键 conversation.id 只兜「全空且同秒」的极端并列。
    """
    rows = db.execute(
        select(Conversation, func.count(Message.id), func.coalesce(func.max(Message.id), 0))
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
        .group_by(Conversation.id)
        .order_by(
            Conversation.updated_at.desc(),
            func.coalesce(func.max(Message.id), 0).desc(),
            Conversation.id.desc(),
        )
    ).all()
    return [(row[0], row[1]) for row in rows]


def get_conversation(db: Session, *, user_id: int, conversation_id: int) -> Conversation | None:
    """按 id 查会话，强制带 user_id 过滤（越权防线与 get_kb_group 同款：
    别人的会话一律查不到，api 层转 404 防探测）。"""
    return db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
    ).scalar_one_or_none()


def list_messages(db: Session, *, user_id: int, conversation_id: int) -> list[Message]:
    """取某会话全部消息（先校验会话归属再取，按 id 升序=时间正序）。

    为什么先 get_conversation：只按 conversation_id 查消息会漏掉归属校验——
    攻击者拿着别人的 conversation_id 直接问消息就能拖走全部历史（越权读）。
    归属不过关返回空列表会让调用方误判成「空会话」，不如由 api 层先 404。
    """
    if get_conversation(db, user_id=user_id, conversation_id=conversation_id) is None:
        return []
    rows = db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    ).scalars()
    return list(rows)


def append_message_pair(
    db: Session,
    *,
    conversation: Conversation,
    question: str,
    answer: str,
    hit: bool,
    sources_json: str,
) -> None:
    """一问一答两条消息原子落库（同一次 commit），并刷新会话 updated_at。

    为什么必须成对原子写：只写成功一半（有问无答/有答无问）会让回看时间线
    错乱；一次 commit 要么整对可见要么都不可见——SQLAlchemy 同 session 事务天然保证。

    为什么手动刷新 conversation.updated_at 而只靠 onupdate：
    onupdate 只在「该行本身被 UPDATE」时触发，追加 Message 不会碰 Conversation 行；
    这里显式赋值（DB 时钟 func.now()）把「会话有新消息」传导成「会话最近活跃」，
    列表倒序才排得对。
    """
    db.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=question,
            hit=None,  # 用户行无命中语义（模型注释：NULL 防读侧误判成兜底）
            sources="[]",
        )
    )
    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            hit=hit,
            sources=sources_json,
        )
    )
    # 显式 UPDATE 触发 updated_at 刷新（func.now()=数据库时钟，与 created_at 同源）
    conversation.updated_at = func.now()
    db.commit()


def delete_conversation(db: Session, *, user_id: int, conversation_id: int) -> bool:
    """删除会话并手工级联删其全部消息；非本人/不存在返回 False。

    为什么手工级联而不只靠外键 ON DELETE CASCADE：单测 SQLite 默认不启用外键
    （与 delete_kb_group 同一原因）——手工「先消息后会话」保证两种存储都不留孤儿行。
    """
    conversation = get_conversation(db, user_id=user_id, conversation_id=conversation_id)
    if conversation is None:
        return False
    db.execute(delete(Message).where(Message.conversation_id == conversation.id))
    db.delete(conversation)
    db.commit()
    return True


# ===== 错题记录（M5）=====


def create_quiz_record(
    db: Session,
    *,
    user_id: int,
    question: str,
    question_type: str,
    user_answer: str,
    correct_answer: str,
    explanation: str,
    is_correct: bool,
    ref_file: str = "",
    ref_page: int = 0,
) -> QuizRecord:
    """落一条错题/答题记录（判分时即时写，零 LLM）。"""
    record = QuizRecord(
        user_id=user_id,
        question=question,
        question_type=question_type,
        user_answer=user_answer,
        correct_answer=correct_answer,
        explanation=explanation,
        is_correct=is_correct,
        ref_file=ref_file,
        ref_page=ref_page,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_quiz_records(
    db: Session, *, user_id: int, limit: int = 50, wrong_only: bool = True
) -> list[QuizRecord]:
    """本人最近答题记录，按时间倒序。

    wrong_only=True（默认）=错题本口径只收错的；
    wrong_only=False=含答对的——周报要看正确率与全貌，不能只看错题盲区。
    """
    stmt = select(QuizRecord).where(QuizRecord.user_id == user_id)
    if wrong_only:
        stmt = stmt.where(QuizRecord.is_correct.is_(False))
    rows = db.execute(stmt.order_by(QuizRecord.id.desc()).limit(limit)).scalars()
    return list(rows)


def delete_quiz_record(db: Session, *, user_id: int, record_id: int) -> bool:
    """删一条错题（带 user_id 过滤，非本人返回 False → api 层转 404）。"""
    record = db.execute(
        select(QuizRecord).where(QuizRecord.id == record_id, QuizRecord.user_id == user_id)
    ).scalar_one_or_none()
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True


# ===== 长效记忆（M5）=====


def create_memory_fact(
    db: Session, *, user_id: int, kind: str, content: str, ref_file: str = ""
) -> MemoryFact:
    """写一条记忆事实（MySQL=事实源；向量双写由调用方 best-effort 补）。"""
    fact = MemoryFact(user_id=user_id, kind=kind, content=content, ref_file=ref_file)
    db.add(fact)
    db.commit()
    db.refresh(fact)
    return fact


def list_memory_facts(
    db: Session, *, user_id: int, limit: int = 20, kinds: tuple[str, ...] | None = None
) -> list[MemoryFact]:
    """本人记忆清单（可按 kind 过滤），时间倒序（最近的排前——越近越相关）。"""
    stmt = select(MemoryFact).where(MemoryFact.user_id == user_id)
    if kinds:
        stmt = stmt.where(MemoryFact.kind.in_(kinds))
    rows = db.execute(
        stmt.order_by(MemoryFact.id.desc()).limit(limit)
    ).scalars()
    return list(rows)


def latest_memory_fact(db: Session, *, user_id: int, kind: str) -> MemoryFact | None:
    """取某类最新一条（GET 最近一次学情报告用）。"""
    return db.execute(
        select(MemoryFact)
        .where(MemoryFact.user_id == user_id, MemoryFact.kind == kind)
        .order_by(MemoryFact.id.desc())
        .limit(1)
    ).scalar_one_or_none()
