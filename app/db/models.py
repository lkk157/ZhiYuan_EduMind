# -*- coding: utf-8 -*-
"""
ORM 表模型。M1 建 users；M2 建知识库三件套（kb_groups / documents / chunk_fingerprints）；
RAG 完善阶段补会话历史两件套（conversations / messages）。

为什么不一次建全 ROADMAP 里的十几张表（日志/记忆/错题……）：
「一个阶段只做一个阶段的事」（CLAUDE.md §2）——空表堆积只会增加维护成本，
后续表随各自功能（M4 滑窗直接复用本文件会话两表、M5 记忆表、M6 日志表）在对应里程碑落地。
所有核心业务表都会带 user_id 外键（生产思维 #4：多端数据一致性靠它隔离）。
"""
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class User(Base):
    """用户表：账号密码登录，user_id 是全库数据归属的锚点。"""

    __tablename__ = "users"

    # 为什么用 with_variant：主键自增在两家数据库的规则不同——
    # MySQL 用 BIGINT AUTO_INCREMENT（生产规范，对应 init_db.sql），
    # 而 SQLite 只有 INTEGER PRIMARY KEY 才是 rowid 别名会自增（BIGINT 不会！），
    # 单测跑 SQLite 时必须退化成 Integer，否则插入报 NOT NULL(id)。
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    # 用户名唯一 + 建索引：登录查询按 username 走索引
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # bcrypt 哈希固定 60 字符，留 255 是给未来换哈希算法留余量
    password_hash: Mapped[str] = mapped_column(String(255))
    # 时间交给数据库生成（CURRENT_TIMESTAMP），避免应用时区不一致的坑
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class KbGroup(Base):
    """知识库分组表：向量库物理隔离的载体。

    为什么要有分组这一层（而不是所有文档堆在一个库）：
    1. 向量检索按「用户 × 分组」建独立 Chroma collection（store_for 的 u{user_id}g{group_id}），
       分组就是这次物理隔离的边界——删分组即删整个 collection，知识按课程/主题互不串味；
    2. 检索时可多选分组跨库合并，比单库加过滤元数据更省显存（collection 天然分片）；
    3. 分组名按用户唯一：同一个人不能建两个同名分组（避免前端列表歧义），不同用户互不影响。
    """

    __tablename__ = "kb_groups"
    __table_args__ = (
        # 唯一键名与 scripts/init_db.sql 一致，避免两套 DDL 漂移后运维排查困难
        UniqueConstraint("user_id", "name", name="uk_kb_groups_user_name"),
    )

    # 主键方言适配与 User.id 同理（原因见上方 with_variant 注释）
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    # 归属人：知识库的一切读写都先过 user_id 过滤（防探测——别人的分组查不到）
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_kb_groups_user"),
    )
    name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<KbGroup id={self.id} user_id={self.user_id} name={self.name!r}>"


class Document(Base):
    """文档表：一个上传文件一行；同名重传 = 同一行的版本更新（走 chunk 级增量）。

    为什么 (user_id, group_id, file_name) 唯一，而不是每次上传都插新行：
    1. 向量 id 采用稳定命名 "{document_id}:{chunk_index}"——document_id 必须跨重传不变，
       同名重传才能 upsert 覆盖同块，未变更块零成本保留（否则向量库会攒出一堆孤儿旧版本）；
    2. 同名重传语义是「这份资料的修订版」而不是「另一份资料」，页面溯源（file_name + page_no）
       在用户视角也始终指向同一个文件名；
    3. file_hash 记录整文件 sha256：内容完全没变的重传直接跳过（skipped_identical），省一轮 OCR/嵌入。
    """

    __tablename__ = "documents"
    __table_args__ = (
        # 唯一键名与 scripts/init_db.sql 一致
        UniqueConstraint("user_id", "group_id", "file_name", name="uk_docs_user_group_name"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_docs_user"),
    )
    group_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("kb_groups.id", ondelete="CASCADE", name="fk_docs_group"),
    )
    file_name: Mapped[str] = mapped_column(String(255))
    # 存字符串而非 JSON/路径对象：DB 要可移植（MySQL/SQLite 同构），代码侧用 pathlib.Path(file_path) 还原
    file_path: Mapped[str] = mapped_column(String(512))
    # 全文件 sha256 十六进制（64 字符）：整文件级别的幂等短路判断
    file_hash: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # 无文本层页码（扫描版 PDF 等）逗号拼接存字符串，编码/还原见 join_empty_pages / split_empty_pages
    empty_pages: Mapped[str] = mapped_column(String(512), default="")
    # ready=可检索；failed=入库流水线中止（解析失败等），前端据此给出「重新上传」提示
    status: Mapped[str] = mapped_column(String(16), default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # updated_at 随每次版本更新刷新：前端列表按它排序能一眼看到「最近改过」的资料
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} group_id={self.group_id} "
            f"file_name={self.file_name!r} status={self.status!r}>"
        )


class ChunkFingerprint(Base):
    """块指纹表：记录「当前有效」的每个文本块的哈希，是 chunk 级增量的比对基线。

    为什么单独一张表（而不是把哈希塞进 documents 一列）：
    1. 增量比对需要 old: dict[chunk_index, chunk_hash] 这个逐块结构，一文档多行天然映射，
       直接喂给 diff_chunks 算 added/changed/removed/unchanged；
    2. page_no 的溯源根基就在块粒度上——每块独立哈希后，块没变则其向量连同 metadata
       （file_name/page_no/chunk_index）原样保留，引用页码才不会随重传漂移；
    3. (document_id, chunk_index) 唯一：同一块只留一条「当前指纹」，
       配合向量 id "{document_id}:{chunk_index}" 的稳定命名，upsert 覆盖同块，
       旧版本块的指纹在 replace_chunk_fingerprints 整组替换时被清掉，不会误判。
    """

    __tablename__ = "chunk_fingerprints"
    __table_args__ = (
        # 唯一键名与 scripts/init_db.sql 一致
        UniqueConstraint("document_id", "chunk_index", name="uk_chunks_doc_index"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("documents.id", ondelete="CASCADE", name="fk_chunks_document"),
    )
    # 与 Chunk.chunk_index 同构：全文档连续（0 起）、绝不跨页，删块后中间可能断号（属正常）
    chunk_index: Mapped[int] = mapped_column(Integer)
    # chunk_sha256(text) 的十六进制结果（64 字符），文本未变则哈希未变
    chunk_hash: Mapped[str] = mapped_column(String(64))

    def __repr__(self) -> str:
        return (
            f"<ChunkFingerprint document_id={self.document_id} "
            f"chunk_index={self.chunk_index}>"
        )


class Conversation(Base):
    """问答会话表：一次连续答疑 = 一行，消息（Message）挂在其下（会话历史的父表）。

    为什么标题由首问截取而不是让 LLM 起标题：历史保存是纯 DB 功能，
    引入一次「LLM 起标题」就把本阶段「零新增模型调用、不碰显存红线」的承诺破坏了；
    会话列表对标题精度要求低，首问前 30 字足够辨识（生成端见前端 create 调用处）。

    为什么没有唯一约束：同名会话（如多个「新对话」）在产品上完全合法——
    这与 kb_groups 的「分组名按用户唯一」语义不同：那边唯一是为了防前端
    name→id 映射歧义；会话列表一律用 id 做键，重名无害，强加唯一反而逼用户改名。

    为什么 updated_at 要随消息追加刷新（onupdate）：会话列表按最近活跃倒序排，
    每次问答都更新父行时间戳，用户打开页面一眼看到「最近在聊的对话」。
    """

    __tablename__ = "conversations"

    # 主键方言适配与 User.id 同理（原因见文件头 with_variant 注释）
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    # 归属人：一切会话读写先过 user_id 过滤（越权防线与分组/文档同款防探测纪律）
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_conversations_user"),
    )
    title: Mapped[str] = mapped_column(String(128), default="新对话")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # onupdate=func.now()：对该行做 UPDATE 时数据库端自动刷新（与 documents 同款），
    # 保证与 created_at 用的是同一个数据库时钟，排序不会混进应用机本地时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} user_id={self.user_id} title={self.title!r}>"


class Message(Base):
    """会话消息表：一问（role=user）一答（role=assistant）各一行。

    为什么 hit/sources 必须落库：回看历史要还原「命中徽章 + 来源卡片」——
    本项目卖点是溯源，历史里丢了来源，回看时卖点就消失了；
    sources 整体 JSON 序列化存字符串列，不拆关系表：问答产物是一次性读写的整体，
    拆表只有坏处，且与 documents.empty_pages 同一取舍（MySQL/SQLite 同构、避开 JSON 方言差异）。

    为什么 hit 可空（NULL）而不是默认 False：用户提问行没有「命中」语义，
    存 NULL 让读侧结构上无法把「用户消息」误判成「兜底回答」（False 是兜底的语义）。

    为什么挂 conversation_id 索引：回看按会话整取消息（WHERE conversation_id=? ORDER BY id），
    会话一多没有索引就是全表扫（生产思维——索引在建表时就位，不等慢了再补）。
    """

    __tablename__ = "messages"
    # 建索引：列表页按会话取消息的唯一查询路径（与 users.username 同思路）
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("conversations.id", ondelete="CASCADE", name="fk_messages_conversation"),
        index=True,
    )
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    # "user" / "assistant"：与前端 chat_message 角色、OpenAI 惯例对齐，禁止发明第三种值
    role: Mapped[str] = mapped_column(String(16))
    # Text 而非 String：答案可达数百字，MySQL 的 VARCHAR 上限与语义都不如 TEXT 贴切
    content: Mapped[str] = mapped_column(Text)
    # None=用户消息（无命中语义）；True/False=助手回答是否命中知识库
    hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    # JSON 字符串（如 [{"file_name":..,"page_no":..,"snippet":..}]），出口用 json.loads 还原
    sources: Mapped[str] = mapped_column(Text, default="[]")
    # 回看时展示时间线；排序仍以 id 为准（自增单调，不受同秒并列影响）
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id} conversation_id={self.conversation_id} "
            f"role={self.role!r}>"
        )


class MemoryFact(Base):
    """长效记忆事实表（M5）：错题直出的结构化弱点 + 周报 LLM 提炼的洞察与报告。

    为什么 kind 要分类：三类的消费方式不同——weak_point 进答疑 prompt 的学情背景、
    report 是用户直接读的周报、insight 是提炼的通用结论；混在一起读侧还得猜语义。

    为什么 ref_file 可空：错题型记忆带章节锚点（供知识点图谱关联推荐回溯），
    报告型记忆没有单一锚点，空串表示无锚——不是所有事实都挂得上章节。
    双写的另一半（向量索引）在 Chroma u{id}mem collection，本表是事实源。
    """

    __tablename__ = "memory_facts"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_memory_user"),
        index=True,
    )
    # weak_point / insight / report（见 long_term.py 的生成端，禁止发明新值）
    kind: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    # 章节锚点（文件名），无锚为空串——知识点图谱推荐的回溯依据
    ref_file: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<MemoryFact id={self.id} user_id={self.user_id} kind={self.kind!r}>"


class QuizRecord(Base):
    """错题记录表（M5）：判分时答错的题各一行——错题本的数据源。

    为什么判分时就落库而不是周报时再挖：对错与解析在判分瞬间全是现成结构
    （零 LLM 即可记录），攒到以后再挖反而丢上下文；错题本页要「即时可见」。

    ref_file/ref_page：本题所依据的课件锚点（来自回答的 sources，前端随判分一并提交）——
    知识点图谱的「错题 → 章节 → 关联推荐」链路全靠它起步，无锚则推荐退化为全局结构边。
    """

    __tablename__ = "quiz_records"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_quiz_records_user"),
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    # choice / short（与试题 JSON 契约一致）
    question_type: Mapped[str] = mapped_column(String(16), default="choice")
    user_answer: Mapped[str] = mapped_column(Text, default="")
    correct_answer: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    ref_file: Mapped[str] = mapped_column(String(255), default="")
    ref_page: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<QuizRecord id={self.id} user_id={self.user_id} correct={self.is_correct}>"


def join_empty_pages(pages: list[int]) -> str:
    """把无文本层页码列表编码成逗号拼接字符串（如 [2, 5] -> "2,5"）。

    为什么 DB 列存字符串而不是 JSON/关联表：
    empty_pages 只是溯源附属信息（每文档少量页码），逗号拼接在 MySQL/SQLite 都能
    原样存储与比较，避开 JSON 方言差异；接口层用 split_empty_pages 还原成 list[int] 契约字段。
    """
    return ",".join(str(p) for p in pages)


def split_empty_pages(text: str | None) -> list[int]:
    """把库里的逗号拼接字符串还原成页码列表（接口返回 empty_pages:list[int] 契约所需）。

    空串/None 都视作「没有无文本层页」——入库前字段默认空串，容错读旧数据。
    """
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


__all__ = [
    "User",
    "KbGroup",
    "Document",
    "ChunkFingerprint",
    "Conversation",
    "Message",
    "MemoryFact",
    "QuizRecord",
    "join_empty_pages",
    "split_empty_pages",
]
