# -*- coding: utf-8 -*-
"""
业务异常与错误码体系。

为什么需要这一层：
1. 前端（Streamlit）要向用户展示「人话」错误提示，而不是裸的 500 栈回溯；
2. 错误码语义化（401 鉴权失败 / 404 不存在 / 409 冲突 / 502 上游服务故障），
   前端可以按码分支处理（比如 401 跳登录页）；
3. 统一异常出口后，业务代码只需 `raise AuthError("xxx")`，不必每处手写 JSON 响应。
"""
import logging

logger = logging.getLogger(__name__)


class AppError(Exception):
    """业务异常基类：带 HTTP 状态码 + 面向用户的提示信息。"""

    code = 500
    message = "服务器内部错误"

    def __init__(self, message: str | None = None, code: int | None = None):
        self.message = message or self.message
        self.code = code or self.code
        super().__init__(self.message)


class AuthError(AppError):
    """鉴权失败：未登录 / token 过期 / 签名伪造。"""

    code = 401
    message = "请先登录"


class NotFoundError(AppError):
    """资源不存在。"""

    code = 404
    message = "资源不存在"


class ConflictError(AppError):
    """资源冲突（如用户名已被注册）。"""

    code = 409
    message = "资源冲突"


class UpstreamError(AppError):
    """上游服务（Ollama/MySQL 等）故障——网络或服务未启动时抛这个。"""

    code = 502
    message = "上游服务暂不可用，请稍后重试"


def install_error_handlers(app) -> None:
    """把业务异常统一转成 JSON 错误响应。

    响应体统一为 {"error": {"code": ..., "message": ...}}，前端一个解析函数走天下。
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError):
        # 401/404/409 属于「预期内」业务错误，记 INFO 即可；5xx 记 ERROR 带栈
        if exc.code >= 500:
            logger.error("业务异常: %s", exc, exc_info=True)
        else:
            logger.info("业务异常: %s", exc)
        return JSONResponse(
            status_code=exc.code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        # 参数校验失败也统一成 {"error":...} 格式（否则前端要多解析一种 FastAPI 默认结构）。
        # 只取第一条错误的提示，避免把整个 pydantic 栈甩给用户。
        first = exc.errors()[0] if exc.errors() else {}
        detail = f"{first.get('loc', ['?'])[-1]}: {first.get('msg', '参数错误')}"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": 422, "message": f"参数校验失败——{detail}"}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception):
        # 兜底：未预期的异常不把栈抛给用户（安全），但服务端日志必须留全栈便于排查
        logger.error("未处理异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "服务器内部错误，请查看日志"}},
        )
