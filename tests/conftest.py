# -*- coding: utf-8 -*-
"""
pytest 全局 fixture（全项目复用的「三大件」之一：数据库）。

为什么在 conftest 里集中造 fixture：
1. 每个测试拿到干净的 SQLite 内存库——零外部依赖、不污染真实 MySQL、跑完即弃；
2. 后续里程碑（M2 Chroma / Mock LLM）也会在这里加 fixture，测试写法统一。
"""
import pytest
from sqlalchemy.orm import Session

from app.db.session import Base, make_engine


@pytest.fixture()
def db_session():
    """干净的 SQLite 内存会话：每个测试独立建表，互不干扰。

    为什么用 make_engine("sqlite:///:memory:")：StaticPool 保证同一连接共享内存库
    （见 app/db/session.py 的注释）。
    """
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture()
def gateway():
    """独立的 OllamaGateway 实例（不共用全局单例的锁状态，测试互不影响）。"""
    from app.core.llm import OllamaGateway

    return OllamaGateway(base_url="http://testserver")
