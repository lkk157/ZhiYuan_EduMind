# -*- coding: utf-8 -*-
"""
前端 HTTP 封装：带 JWT 调 FastAPI，统一解析业务错误。

为什么独立成 services 层（而不是页面里散落 httpx 调用）：
1. 后端错误统一是 {"error":{"code","message"}}（app/core/exceptions.py 的出口约定），
   解析集中在这里，页面只捕获 ApiError 拿人话提示，不必每处写 try/JSON 判断；
2. Bearer 令牌注入只在一个地方——漏带 token 的低级错从结构上杜绝；
3. 与 Streamlit 零耦合：本模块可被脚本/测试直接 import（demo_e2e.py 同一思路），
   体现「前端只走 HTTP 的多端架构」（答辩可讲）。

为什么不把 httpx 客户端塞进 @st.cache_resource 常驻：
cache_resource 是**全进程共享**的——多标签页/多用户会共用同一个对象，
把 token 放进去等于跨用户串号。ApiClient 每次脚本重跑现建只花几毫秒，
用完即弃，安全远比省一次握手重要（session-state 参考文档的同款告诫）。
"""
import os

import httpx

# 后端地址：环境变量 API_BASE_URL 可覆盖（多端演示：换一个地址就指向另一台后端）
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# 上传/问答要等入库向量化或 7B 生成，默认 120s 不够，单独放宽到 300s
_LONG_TIMEOUT = 300.0


class ApiError(Exception):
    """业务错误（后端 {"error":{"code","message"}} 统一出口的镜像），message 是人话提示。"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class ApiClient:
    """知源后端轻客户端：鉴权头注入 + 错误翻译。token 为 None 时打无需登录的接口（注册/登录）。"""

    def __init__(self, token: str | None = None, base_url: str = API_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ---------- 基础请求 ----------

    def _headers(self) -> dict:
        # 标准格式：Authorization: Bearer <token>（与后端 get_current_user 对齐）
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs):
        """发请求并把 {"error":{...}} 翻译成 ApiError；成功返回 JSON（无体返回 None）。"""
        try:
            resp = httpx.request(
                method,
                self.base_url + path,
                headers=self._headers(),
                timeout=timeout or self.timeout,
                **kwargs,
            )
        except httpx.HTTPError as e:
            # 连不上后端是最常见的部署问题（uvicorn 没起），提示直接给出排查方向
            raise ApiError(0, "无法连接后端服务（uvicorn 是否已启动？）") from e

        if resp.status_code >= 400:
            detail = None
            try:
                detail = resp.json().get("error")
            except Exception:
                detail = None  # 非统一格式（如网关 502 HTML）走兜底提示
            if isinstance(detail, dict) and detail.get("message"):
                raise ApiError(detail.get("code", resp.status_code), detail["message"])
            raise ApiError(resp.status_code, f"请求失败（HTTP {resp.status_code}）")

        if not resp.content:
            return None
        return resp.json()

    # ---------- 鉴权 ----------

    def register(self, username: str, password: str) -> dict:
        return self._request("POST", "/auth/register", json={"username": username, "password": password})

    def login(self, username: str, password: str) -> dict:
        """登录成功后顺手把令牌存到实例上——调用方不必手工搬运 access_token。"""
        data = self._request("POST", "/auth/login", json={"username": username, "password": password})
        self.token = data.get("access_token")
        return data

    # ---------- 知识库 ----------

    def list_groups(self) -> list:
        return self._request("GET", "/kb/groups")

    def create_group(self, name: str) -> dict:
        return self._request("POST", "/kb/groups", json={"name": name})

    def delete_group(self, group_id: int) -> dict:
        return self._request("DELETE", f"/kb/groups/{group_id}")

    def list_documents(self, group_id: int) -> list:
        return self._request("GET", f"/kb/groups/{group_id}/documents")

    def upload_documents(self, group_id: int, files: list[tuple[str, bytes, str]]) -> dict:
        """批量上传并入库。files=[(文件名, 字节, content_type), ...]，一次请求一批。

        multipart 用「同名字段列表」形式：多个 ("file", ...) 对 → 后端 list[UploadFile]。
        返回 {results:[逐文件结果], succeeded, failed}；入库含解析+向量化（可能带 OCR），
        耗时可达分钟级，超时单独放宽到 300s。
        """
        parts = [("file", (name, data, ctype or "application/octet-stream")) for name, data, ctype in files]
        return self._request(
            "POST",
            f"/kb/groups/{group_id}/documents",
            files=parts,
            timeout=_LONG_TIMEOUT,
        )

    def delete_document(self, doc_id: int) -> dict:
        return self._request("DELETE", f"/kb/documents/{doc_id}")

    # ---------- 问答 ----------

    def ask(self, question: str, group_ids: list[int] | None = None) -> dict:
        """提问。group_ids=None=检索本人全部分组；返回 {answer, sources, hit}。"""
        return self._request(
            "POST",
            "/chat/ask",
            json={"question": question, "group_ids": group_ids},
            timeout=_LONG_TIMEOUT,  # 7B 生成可能到分钟级（含模型冷加载）
        )
