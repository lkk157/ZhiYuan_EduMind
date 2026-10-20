# -*- coding: utf-8 -*-
"""
M7-1 单测：容器存活探针 /healthz——免鉴权、不碰数据库与模型。

为什么单独一个文件而不是塞进 test_smoke：
test_smoke 锁定的是「项目骨架文件」，本用例锁定的是「编排契约」——
容器 healthcheck 会高频调用它，鉴权或依赖耦合任何一点变化都必须被测试立刻抓住。
"""
from fastapi.testclient import TestClient


def test_healthz_is_public_and_dependency_free():
    """无令牌 200、无 DB 依赖：裸构造 TestClient（不触发 startup、不做 dependency override）。

    裸构造正是关键：若实现里误加了 Depends(get_current_user) 或查库依赖，
    没有令牌的请求会 401/500，而正常路径 200——测试直接锁死契约。
    """
    from app.main import app

    client = TestClient(app)  # 裸构造：不触发 startup（不连真 MySQL）
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
