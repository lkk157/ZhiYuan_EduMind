# -*- coding: utf-8 -*-
"""
数据读写封装（CRUD）：api 层不直接写 SQL，统一走这里。

为什么要有这一层：
1. SQL 散落在接口里会和参数校验/错误处理缠成一团，难测难改；
2. 唯一约束冲突这类数据库细节在这里翻译成业务异常（ConflictError），
   接口层只管用人话回给前端。
"""
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError
from app.db.models import User


def create_user(db: Session, username: str, password_hash: str) -> User:
    """创建用户。用户名撞车时抛 ConflictError（而不是泄漏数据库异常细节）。"""
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()  # 冲突后必须回滚，否则会话进入坏状态后续操作全报错
        # 只把「唯一约束冲突」翻译成用户名已注册；其他完整性错误（缺字段等）必须
        # 如实上抛——否则真 bug 会被误报成业务冲突（M1 单测实测踩过这个坑）。
        # SQLite 报 "UNIQUE constraint failed"，MySQL 报 "Duplicate entry"。
        detail = str(getattr(e, "orig", e)).lower()
        if "unique" in detail or "duplicate" in detail:
            raise ConflictError("用户名已被注册") from e
        raise
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    """按用户名查用户（登录用）。"""
    return db.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """按 id 查用户（JWT 解析后回查，确认账号仍存在且归属正确）。"""
    return db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
