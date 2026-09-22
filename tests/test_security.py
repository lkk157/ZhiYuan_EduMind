# -*- coding: utf-8 -*-
"""
M1 单测：密码哈希 + JWT 签发/校验。

为什么先测这里：security 是「数据归属 user_id」的根基，
这里出漏洞整个多端鉴权都是纸糊的——所以伪造签名/过期令牌必须有自动化守卫。
"""
import pytest

from app.core.exceptions import AuthError
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_and_verify():
    """正确密码通过、错误密码拒绝——bcrypt 哈希的基本盘。"""
    hashed = hash_password("secret123")
    assert hashed != "secret123"  # 绝不存明文
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong-pass", hashed) is False


def test_password_hash_is_salted():
    """同一密码两次哈希结果不同（自带随机盐）——防止彩虹表撞库。"""
    assert hash_password("same-pass") != hash_password("same-pass")


def test_jwt_roundtrip():
    """签发 → 解析往返：载荷里的 user_id/username 必须原样取回。"""
    token = create_access_token(user_id=42, username="demo")
    payload = decode_access_token(token)
    assert payload["user_id"] == 42
    assert payload["username"] == "demo"


def test_jwt_expired_rejected():
    """过期令牌必须被拒绝（用负的过期分钟数立刻造一个过期 token）。"""
    from datetime import datetime, timedelta, timezone

    import jwt as pyjwt
    from app.core.config import settings

    payload = {
        "user_id": 1,
        "username": "old",
        "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    expired = pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_access_token(expired)


def test_jwt_tampered_signature_rejected():
    """伪造签名（篡改载荷后重签）必须被拒绝——多端鉴权的安全底线。"""
    token = create_access_token(user_id=1, username="demo")
    # 篡改：把 payload 段换成别人的 user_id（保持原签名 → 签名校验必失败）
    head, body, sig = token.split(".")
    import base64

    fake_body = base64.urlsafe_b64encode(b'{"user_id":999,"username":"hacker"}').decode().rstrip("=")
    tampered = f"{head}.{fake_body}.{sig}"
    with pytest.raises(AuthError):
        decode_access_token(tampered)


def test_jwt_garbage_rejected():
    """乱码字符串也要走 AuthError 人话路径，不能泄漏异常栈。"""
    with pytest.raises(AuthError):
        decode_access_token("not-a-jwt-token")
