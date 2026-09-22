# -*- coding: utf-8 -*-
"""
鉴权接口：注册 / 登录 / 当前用户解析（JWT）。

为什么接口层这么薄（CLAUDE.md 分层约定）：
只做「参数校验 → 调 crud/security → 返回」，业务算法一概不写——
这样改接口不会碰坏安全逻辑，安全逻辑也能被单测直接测。
"""
from fastapi import APIRouter, Depends, Header, status
from pydantic import BaseModel, Field

from app.core.exceptions import AuthError
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.db import crud
from app.db.models import User
from app.db.session import get_db

# 路由前缀 /auth：所有鉴权相关接口都挂在它下面
router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """注册入参。长度约束做基础防滥用（正式产品还会加频控——生产思维可讲）。"""

    username: str = Field(min_length=3, max_length=32, description="用户名")
    password: str = Field(min_length=6, max_length=64, description="密码")


class LoginRequest(BaseModel):
    """登录入参。"""

    username: str
    password: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db=Depends(get_db)):
    """注册新用户。密码经 bcrypt 哈希后落库（绝不存明文）。"""
    user = crud.create_user(db, username=body.username, password_hash=hash_password(body.password))
    return {"id": user.id, "username": user.username}


@router.post("/login")
def login(body: LoginRequest, db=Depends(get_db)):
    """登录成功签发 JWT。

    为什么失败提示统一是「用户名或密码错误」：不区分「用户不存在/密码错」，
    防止攻击者枚举出哪些用户名已注册（信息安全的基本功，答辩可讲）。
    """
    user = crud.get_user_by_username(db, body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise AuthError("用户名或密码错误")

    token = create_access_token(user_id=user.id, username=user.username)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "username": user.username}


def get_current_user(authorization: str | None = Header(None), db=Depends(get_db)) -> User:
    """FastAPI 依赖：从 Authorization 头解析 JWT → 回查数据库得当前用户。

    为什么解析后还要回查数据库（而不是只信 token）：
    token 有效期内用户可能已被删除；回查保证「数据归属锚点 user_id」真实存在。
    用法：在任意需要登录的接口参数里加 Depends(get_current_user)。

    为什么 Header 声明为可选而不是 Header(...)：必填声明会让 FastAPI 在进业务代码
    前就抛 422 默认格式，前端要多解析一种错误结构；声明为可选、缺失时抛 AuthError，
    全部鉴权失败都走统一出口 {"error":{"code":401,...}}（M1 真机冒烟实测的体验问题）。
    """
    # 标准格式：Authorization: Bearer <token>
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("请携带 Bearer 令牌")

    payload = decode_access_token(token)
    user = crud.get_user_by_id(db, int(payload["user_id"]))
    if user is None:
        raise AuthError("用户不存在或已注销")
    return user
