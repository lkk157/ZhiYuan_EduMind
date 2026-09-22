# -*- coding: utf-8 -*-
"""
SQLAlchemy 连接与会话管理（MySQL = 事实源）。

为什么这么设计：
1. 生产连 MySQL、单测切 SQLite 内存库——同一套 ORM 两套存储，
   单测因此零外部依赖、秒级跑完（conftest 里用 make_engine("sqlite:///:memory:")）；
2. MySQL 连接加 pool_pre_ping + pool_recycle：长时间没人访问时 MySQL 会主动断连
   （wait_timeout），不加健康检查下一次查询就报 "MySQL server has gone away"，
   这是长跑服务的经典坑。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings


class Base(DeclarativeBase):
    """ORM 声明基类：所有表模型继承它。"""


def make_engine(url: str):
    """按 URL 造 engine（测试与生产共用的工厂函数）。

    SQLite 特判的原因：内存库 :memory: 默认每个连接各建一个空库，
    必须用 StaticPool 让所有会话共享同一个连接/同一份内存库。
    """
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


# 生产 engine / 会话工厂（从 settings 取连接串，DATABASE_URL 可覆盖）
engine = make_engine(settings.get_database_url())
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI 依赖注入：每个请求一个会话，请求结束自动关闭。

    为什么 expire_on_commit=False：提交后对象还能继续读属性，
    不然一提交就触发懒加载报错（新手最容易踩的 SQLAlchemy 坑之一）。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(bind_engine=None) -> None:
    """幂等建表（create_all 已存在的表会跳过）。

    M1 阶段用它足够；生产演进方向是 Alembic 版本化迁移（README 升级路径）。
    """
    # 建表前必须先 import 模型，否则 Base.metadata 是空的
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind_engine or engine)
