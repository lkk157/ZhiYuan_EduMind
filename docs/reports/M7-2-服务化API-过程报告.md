# M7-2 过程报告 · RAG 服务化 API v1

> 所属阶段：M7-2（服务化里程碑下半程）｜完成日期：2026-10-20｜单测：201 passed（192 → 201，+9）

---

## 一、这阶段做了什么

1. **API Key 体系**：
   - `api_keys` 表（`key_hash` 唯一索引、`key_prefix` 展示、`is_active` 软吊销、`last_used_at`）；
   - `security.py` 生成/哈希：`secrets.token_urlsafe(32)` + sha256（★不用 bcrypt，理由见模型注释：高熵随机串可索引直查，bcrypt 每请求全表 verify 不可接受）；
   - 管理接口（JWT）：`POST/GET/DELETE /api-keys`——明文只返回一次，列表不含明文/哈希，越权 404，重复吊销 404。
2. **/v1 服务化网关**（`app/api/v1.py`）：
   - 鉴权依赖 `X-API-Key` → 哈希查表 → 账号（与 JWT 路径同构，沿用全部 user_id 隔离）；
   - 端点：`GET/POST /v1/kb/groups`、`GET/POST /v1/kb/groups/{gid}/documents`、`POST /v1/rag/retrieve`、`POST /v1/rag/ask`、`POST /v1/rag/ask/stream`（SSE 同事件契约）；
   - **复用而不复制**：分组校验/问答上下文/入库流水线分别调用 `chat.resolve_group_ids`、`chat.prepare_context`、`kb.ingest_uploads`（本阶段把 `/chat/ask` 与 `/kb` 上传里的逻辑抽成公共函数，行为不变，既有 192 项测试保护）。
3. **背压（429）**：`/v1` 容量闸门固定 1（与 gateway 推理锁同源）；排队超过 `V1_QUEUE_TIMEOUT`（默认 30s）→ JSON 端点 `429 + Retry-After: 5`；SSE 端点以 error 事件表达（流已开始，HTTP 头无法再改）。闸门在 SSE 生成器内部 acquire/finally release——避免客户端断开导致容量永久泄漏。
4. **文档**：`docs/API-服务化调用指南.md`（能力边界、签发流程、端点/事件契约、curl/Python 示例、错误码、单并发容量声明）；README 增加服务化章节。
5. **异常体系扩展**：`AppError` 支持 `headers`（429 的 Retry-After），新增 `TooManyRequestsError`。

## 二、遇到了什么问题

1. **★追加 ApiKey 模型时锚文本过短，吞掉了 QuizRecord 的 `created_at` 与 `__repr__`**：Edit 只匹配到 `ref_page` 三行，后续两段代码被新类「接管」；ApiKey 里还出现重复 `created_at`。单跑 test_quiz_records 通过、全量才红（顺序相关表象），险些误判为测试污染；
2. `/v1/rag/ask/stream` 单测收到 error 事件：测试桩只有 `generate`，而流式路径调用 `gateway.generate_stream`（AttributeError）；
3. 重复吊销的语义选择：软删后再次 DELETE 若不校验 `is_active` 会返回 200，与「资源不存在即 404」的既有防探测口径不一致；
4. 原生 8000 服务已停止（产品经理已切到容器），无法直接做原生真机验证。

## 三、怎么解决的问题（与上一一对应）

1. 恢复 QuizRecord 的 `created_at`/`__repr__`，ApiKey 只保留一份 `created_at` 并给正确的 `__repr__`；教训固化——**追加类的 Edit 要把 old_string 延伸到文件末尾或下一段起点**，改完立刻跑全量（本次正是全量抓住的）；
2. 测试桩补 `generate_stream`（逐字回吐），SSE 测试拿到 done 事件；
3. `revoke_api_key` 查询条件加 `is_active=True`：已吊销视为不存在 → 重复 DELETE 404（幂等口径与 quiz 删除一致）；
4. 改用容器（18000 端口）做真机验证，同时顺带证明新镜像内服务化可用。

## 四、真机验证（容器 `zhiyuan-edumind:latest`，2026-10-20）

| 步骤 | 结果 |
|---|---|
| JWT 登录 → `POST /api-keys` | 201（响应含 `zy_` 明文，数据库只有哈希） |
| `GET /v1/kb/groups`（X-API-Key） | 200 |
| `POST /v1/kb/groups` + `POST .../documents` | 200，入库 1 个文档 |
| `POST /v1/rag/retrieve` | 命中 1 条，返回文件/页码/分数 |
| `POST /v1/rag/ask` | 200，`hit=true`、`intent=qa`、来源 1 条（真 LLM） |
| 无 Key / 伪造 Key | 401 / 401 |

**单测终态 201 passed**（新增 9 项：API Key 管理 3、/v1 鉴权/隔离/检索问答/空召回零调用/SSE/429 共 6）。

## 五、边界与遗留

- `/v1` 为无状态单轮（多轮上下文由调用方维护）；如需服务端会话，可在 v2 增加 `conversation_id` 语义；
- 未开放出题/判分/周报等应用层能力（服务化定位是「知识库 + RAG」核心）；
- 多副本水平扩展明确不做：单 GPU 物理上限，且会破坏进程内推理锁；
- Key 的 `last_used_at` 每请求更新（单机低并发可接受；高并发场景应改为节流/异步写）。
