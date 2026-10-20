# -*- coding: utf-8 -*-
"""
M7-2 单测：API Key 管理——明文只出现一次、哈希落库、吊销、越权 404。

为什么用 JWT 管理接口而非直接造 crud 数据：本组测的是「人机接口契约」，
包括响应里有明文、列表里没有——这些只有走 HTTP 才验证得到。
"""
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token, hash_api_key
from app.db import crud
from app.db.session import get_db


@pytest.fixture()
def env(db_session):
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    user_a = crud.create_user(db_session, username="alice", password_hash="x")
    user_b = crud.create_user(db_session, username="bob", password_hash="x")
    client = TestClient(app)  # 裸构造：不触发 startup
    yield {
        "client": client,
        "db": db_session,
        "token_a": create_access_token(user_id=user_a.id, username=user_a.username),
        "token_b": create_access_token(user_id=user_b.id, username=user_b.username),
        "user_a": user_a,
    }
    app.dependency_overrides.clear()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_create_list_revoke_flow(env):
    """签发 → 明文仅响应一次且前缀可识别 → 列表无明文/无哈希 → 吊销后状态翻转。"""
    c = env["client"]
    tok = _auth(env["token_a"])

    r = c.post("/api-keys", json={"name": "课程门户"}, headers=tok)
    assert r.status_code == 201
    body = r.json()
    plain = body["key"]
    assert plain.startswith("zy_") and len(plain) > 30
    assert body["key_prefix"] == plain[: len("zy_") + 8]
    assert body["name"] == "课程门户"

    # 数据库里只有 sha256，没有明文
    rows = crud.list_api_keys(env["db"], user_id=env["user_a"].id)
    assert rows[0].key_hash == hash_api_key(plain)
    assert plain not in rows[0].key_hash

    # 列表契约：不含明文、不含哈希
    items = c.get("/api-keys", headers=tok).json()
    assert len(items) == 1 and items[0]["is_active"] is True
    assert "key" not in items[0] and "key_hash" not in items[0]

    # 吊销（软删）→ 状态 False，行仍在
    assert c.delete(f"/api-keys/{body['id']}", headers=tok).status_code == 200
    items = c.get("/api-keys", headers=tok).json()
    assert items[0]["is_active"] is False
    assert len(crud.list_api_keys(env["db"], user_id=env["user_a"].id)) == 1


def test_management_requires_login(env):
    """管理接口是 JWT 领地：无令牌一律 401（Key 不能自己签 Key）。"""
    assert env["client"].post("/api-keys", json={"name": "x"}).status_code == 401
    assert env["client"].get("/api-keys").status_code == 401


def test_cross_user_revoke_is_404(env):
    """B 吊销 A 的 Key → 404 防探测；A 自己吊销成功；重复吊销仍 404。"""
    c = env["client"]
    kid = c.post("/api-keys", json={"name": "A的"}, headers=_auth(env["token_a"])).json()["id"]
    assert c.delete(f"/api-keys/{kid}", headers=_auth(env["token_b"])).status_code == 404
    assert c.delete(f"/api-keys/{kid}", headers=_auth(env["token_a"])).status_code == 200
    assert c.delete(f"/api-keys/{kid}", headers=_auth(env["token_a"])).status_code == 404
