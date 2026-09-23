# -*- coding: utf-8 -*-
"""
ORM 表模型。M1 建 users；M2 建知识库三件套（kb_groups / documents / chunk_fingerprints）。

为什么不一次建全 ROADMAP 里的十几张表（消息/日志/记忆……）：
「一个阶段只做一个阶段的事」（CLAUDE.md §2）——空表堆积只会增加维护成本，
后续表随各自功能（M3 消息表、M7 记忆表）在对应里程碑落地。
所有核心业务表都会带 user_id 外键（生产思维 #4：多端数据一致性靠它隔离）。
"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
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
    "join_empty_pages",
    "split_empty_pages",
]
