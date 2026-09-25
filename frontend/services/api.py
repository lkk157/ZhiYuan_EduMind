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
import json
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

    def ask(
        self,
        question: str,
        group_ids: list[int] | None = None,
        conversation_id: int | None = None,
        guide_mode: bool = False,
    ) -> dict:
        """提问。group_ids=None=检索本人全部分组；
        返回 {answer, sources, hit, intent, conversation_id}（intent=M4 意图路由标签）。

        conversation_id=None 时后端不落库（无状态，兼容旧行为）；
        传了则一问一答写入该会话（回看历史靠它）；
        guide_mode=True 走引导式答疑（苏格拉底模式，M4）。
        """
        return self._request(
            "POST",
            "/chat/ask",
            json={
                "question": question,
                "group_ids": group_ids,
                "conversation_id": conversation_id,
                "guide_mode": guide_mode,
            },
            timeout=_LONG_TIMEOUT,  # 7B 生成可能到分钟级（含模型冷加载）
        )

    def score_quiz(self, quiz: dict, answers: list[str]) -> dict:
        """试题判分：quiz=出题返回的试题 JSON 对象，answers=按题序作答；
        返回 {score, comment}。单选全卷后端零 LLM 直接算，简答走模型按要点给分。"""
        return self._request("POST", "/chat/score", json={"quiz": quiz, "answers": answers})

    def ask_stream(
        self,
        question: str,
        group_ids: list[int] | None = None,
        conversation_id: int | None = None,
        guide_mode: bool = False,
    ):
        """SSE 流式提问（生成器）：逐个产出事件 dict——
        {"t":"delta","v":"字"} / {"t":"done", answer,sources,hit,intent,…} / {"t":"error","message"}。

        为什么单独开这个方法而不是改 ask：ask 的 JSON 契约有 demo_e2e 与
        大量单测依赖，流式是新增能力（后端 stream=true 分支），两条路各走各的。
        连接层错误照旧翻译成 ApiError；协议内 error 事件原样交调用方处理
        （调用方要区分「已流出半截字」与「压根没开始」两种现场）。
        """
        payload = {
            "question": question,
            "group_ids": group_ids,
            "conversation_id": conversation_id,
            "guide_mode": guide_mode,
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST",
                self.base_url + "/chat/ask",
                json=payload,
                headers=self._headers(),
                timeout=_LONG_TIMEOUT,  # 整段流可能持续到生成结束
            ) as resp:
                if resp.status_code >= 400:
                    # 流开始前的错误（404 越权等）：非 SSE，按统一错误出口解析
                    body_bytes = resp.read()
                    detail = None
                    try:
                        detail = json.loads(body_bytes).get("error")
                    except Exception:
                        detail = None
                    if isinstance(detail, dict) and detail.get("message"):
                        raise ApiError(detail.get("code", resp.status_code), detail["message"])
                    raise ApiError(resp.status_code, f"请求失败（HTTP {resp.status_code}）")
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        yield json.loads(line[len("data: "):])
        except httpx.HTTPError as e:
            raise ApiError(0, "无法连接后端服务（uvicorn 是否已启动？）") from e

    def fetch_page_image(self, group_id: int, file_name: str, page_no: int) -> bytes:
        """取文档某页的 PNG 字节（来源卡片「查看原页」用）。失败抛 ApiError。"""
        resp = httpx.get(
            self.base_url + f"/kb/groups/{group_id}/page-image",
            params={"file_name": file_name, "page_no": page_no},
            headers=self._headers(),
            timeout=60.0,  # PDF 首次渲染可能略慢（本机回环通常几十 ms）
        )
        if resp.status_code >= 400:
            detail = None
            try:
                detail = resp.json().get("error")
            except Exception:
                detail = None
            if isinstance(detail, dict) and detail.get("message"):
                raise ApiError(detail.get("code", resp.status_code), detail["message"])
            raise ApiError(resp.status_code, f"请求失败（HTTP {resp.status_code}）")
        return resp.content

    # ---------- 会话历史 ----------

    def create_conversation(self, title: str) -> dict:
        """新建空会话，返回 {id, title}（前端首问前调用，标题=首问截 30 字）。"""
        return self._request("POST", "/chat/conversations", json={"title": title})

    def list_conversations(self) -> list:
        """列本人全部会话（含 message_count，按最近活跃倒序）——侧边栏数据源。"""
        return self._request("GET", "/chat/conversations")

    def list_messages(self, conversation_id: int) -> list:
        """取某会话全部消息（时间正序），用于切换会话时回看渲染。"""
        return self._request("GET", f"/chat/conversations/{conversation_id}/messages")

    def delete_conversation(self, conversation_id: int) -> dict:
        """删除会话（服务端级联删消息）。"""
        return self._request("DELETE", f"/chat/conversations/{conversation_id}")
