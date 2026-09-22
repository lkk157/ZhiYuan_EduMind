# -*- coding: utf-8 -*-
"""
ORM 表模型。M1 只建 users 表。

为什么不一次建全 ROADMAP 里的十几张表（文档/消息/日志……）：
「一个阶段只做一个阶段的事」（CLAUDE.md §2）——空表堆积只会增加维护成本，
后续表随各自功能（M2 文档表、M3 消息表、M7 记忆表）在对应里程碑落地。
所有核心业务表都会带 user_id 外键（生产思维 #4：多端数据一致性靠它隔离）。
"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, func
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
