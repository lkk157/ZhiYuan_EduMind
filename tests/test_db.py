# -*- coding: utf-8 -*-
"""
M1 单测：用户表 CRUD roundtrip（SQLite 内存库）。

为什么测数据层：user_id 是全库数据归属的锚点（生产思维 #4），
用户创建/查询/唯一约束必须有守卫，后面所有业务表都挂在 user_id 上。
"""
import pytest

from app.core.exceptions import ConflictError
from app.db import crud


def test_user_create_and_fetch(db_session):
    """创建 → 按用户名/按 id 查询，三处字段一致。"""
    user = crud.create_user(db_session, username="demo", password_hash="hash-xxx")
    assert user.id is not None

    by_name = crud.get_user_by_username(db_session, "demo")
    by_id = crud.get_user_by_id(db_session, user.id)
    assert by_name is not None and by_name.id == user.id
    assert by_id is not None and by_id.username == "demo"
    assert by_id.password_hash == "hash-xxx"


def test_duplicate_username_conflict(db_session):
    """用户名唯一约束：重复注册必须抛 ConflictError（409 人话），而不是裸数据库异常。"""
    crud.create_user(db_session, username="demo", password_hash="h1")
    with pytest.raises(ConflictError):
        crud.create_user(db_session, username="demo", password_hash="h2")


def test_missing_user_returns_none(db_session):
    """查不存在的用户返回 None（登录接口据此给出统一的「用户名或密码错误」）。"""
    assert crud.get_user_by_username(db_session, "ghost") is None
    assert crud.get_user_by_id(db_session, 99999) is None
