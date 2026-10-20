# 知源 RAG 服务化 API · 调用指南（v1）

> 受众：需要接入知源知识库/问答能力的其他项目开发者。
> 版本：`/v1`（M7-2，2026-10-20）｜协议：HTTP + JSON / SSE

## 1. 能力与边界（先读这一节）

| 项 | 说明 |
|---|---|
| 提供的能力 | 知识库分组与文档管理、纯检索（切片+页码+分数）、RAG 问答（同步/流式） |
| 鉴权 | `X-API-Key` 请求头（一把 Key 绑定一个系统账号，数据与其他账号完全隔离） |
| **并发容量** | **同一时刻只处理 1 个推理请求**（8G 显存单机 Ollama 的物理上限）。并发调用会排队；排队超过 `V1_QUEUE_TIMEOUT`（默认 30 秒）返回 **429**，请按 `Retry-After` 退避 |
| 问答形态 | **无状态单轮**：服务端不保存调用方的会话；多轮上下文请调用方自行维护 |
| 生成能力 | 问答底层是同一套 Agent（答疑/出题/总结/计算意图自动路由），有课件依据才回答，否则返回兜底话术且 `hit=false` |
| 版本 | 路径前缀 `/v1`；不兼容变更会升 `/v2`，`/v1` 在迁移期内保留 |

> 该服务面向内部低并发集成；要高并发请先升级推理层（独立 GPU/vLLM），不是调大参数能解决的。

## 2. 签发 API Key

管理动作使用**登录后的 JWT**（人操作一次），调用动作使用 **API Key**（机器长期使用）。

```bash
# 1) 登录拿 JWT
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "你的密码"}'
# → {"access_token": "...", ...}

# 2) 签发 Key（响应里的 key 只出现这一次，请立即保存到调用方的密钥管理里）
curl -X POST http://127.0.0.1:8000/api-keys \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"name": "课程门户"}'
# → {"id":1,"name":"课程门户","key":"zy_xxx...","key_prefix":"zy_AbCdEfGh",...}

# 3) 查看 / 吊销（吊销后立即失效，重复吊销返回 404）
curl http://127.0.0.1:8000/api-keys -H "Authorization: Bearer <JWT>"
curl -X DELETE http://127.0.0.1:8000/api-keys/1 -H "Authorization: Bearer <JWT>"
```

## 3. 接口一览

所有 `/v1` 请求都必须带 `X-API-Key: zy_...`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/kb/groups` | 分组列表（含 `doc_count`） |
| POST | `/v1/kb/groups` | 建分组 `{"name": "..."}`（同名 409） |
| GET | `/v1/kb/groups/{gid}/documents` | 文档列表（状态/页数/切块数） |
| POST | `/v1/kb/groups/{gid}/documents` | 批量上传（multipart，字段名 `file`，单批 ≤5 个/≤200MB；支持 PDF/Word/PPT/图片） |
| POST | `/v1/rag/retrieve` | 纯检索（不生成） |
| POST | `/v1/rag/ask` | 同步问答 |
| POST | `/v1/rag/ask/stream` | SSE 流式问答 |
| GET | `/healthz` | 存活探针（免鉴权） |

### 3.1 纯检索

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/retrieve \
  -H "X-API-Key: zy_xxx" -H "Content-Type: application/json" \
  -d '{"query": "单链表的插入时间复杂度", "group_ids": [1], "top_k": 5}'
```

```json
{
  "hits": [
    {"file_name": "2.3.1_单链表的定义.pdf", "page_no": 3,
     "text": "单链表插入需要先找到前驱结点……", "score": 0.83, "group_id": 1}
  ],
  "count": 1
}
```

- `group_ids` 省略 = 账号下全部分组；传了不属于本账号的 id → 404；
- `top_k` 范围 1–20（默认取服务端配置）；
- 打桩建议：把 `hits[].text` 连同页码交给自己的生成模型时，务必保留出处，便于溯源。

### 3.2 同步问答

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/ask \
  -H "X-API-Key: zy_xxx" -H "Content-Type: application/json" \
  -d '{"question": "学习率过大会怎样？", "group_ids": [1]}'
```

```json
{
  "answer": "学习率过大时损失会在最优点附近震荡……【来源：讲义.pdf，第1页】",
  "sources": [{"file_name": "讲义.pdf", "page_no": 1, "snippet": "...", "score": 0.82}],
  "hit": true,
  "intent": "qa",
  "scope_note": null
}
```

- `hit=false` 表示知识库无依据，`answer` 是固定兜底话术、`sources=[]`——请据此向你的用户展示「未找到」，**不要**把它当成正常答案；
- `intent ∈ qa/quiz/summary/calc`（服务端自动路由）；
- `guide_mode=true` 可切换为引导式答疑（苏格拉底模式）。

### 3.3 流式问答（长回答推荐）

```bash
curl -N -X POST http://127.0.0.1:8000/v1/rag/ask/stream \
  -H "X-API-Key: zy_xxx" -H "Content-Type: application/json" \
  -d '{"question": "总结第一章", "group_ids": [1]}'
```

事件契约（每行 `data: <json>`）：

| `t` | 字段 | 含义 |
|---|---|---|
| `delta` | `v` | 增量文本（逐字/逐段） |
| `done` | `answer/sources/hit/intent/scope_note` | 终局结果（与同步问答一致） |
| `error` | `message` | 流内失败（含排队超时）——收到后不要再等 `done` |

## 4. Python 调用示例

```python
import httpx

API = "http://127.0.0.1:8000"
KEY = {"X-API-Key": "zy_xxx"}

with httpx.Client(base_url=API, headers=KEY, timeout=300) as c:
    # 纯检索：自己拼 prompt
    hits = c.post("/v1/rag/retrieve",
                  json={"query": "什么是线性表", "group_ids": [1]}).json()["hits"]

    # 问答：拿答案 + 出处
    r = c.post("/v1/rag/ask", json={"question": "什么是线性表？", "group_ids": [1]})
    body = r.json()
    if body["hit"]:
        print(body["answer"])
        for s in body["sources"]:
            print(f"- {s['file_name']} 第{s['page_no']}页")
    else:
        print("知识库中没有依据")
```

## 5. 错误码

| HTTP | 场景 | 建议处理 |
|---|---|---|
| 401 | 缺 `X-API-Key` / Key 无效或已吊销 | 检查配置或重新签发 |
| 404 | 分组/文档不存在或不属于本账号 | 不要重试，先核对 group_id |
| 409 | 分组重名 | 换名或复用已有分组 |
| 422 | 参数校验失败（如空问题、top_k 超限） | 修正请求 |
| **429** | 推理容量已满、排队超时 | 按 `Retry-After` 退避后重试（默认 5 秒） |
| 502 | Ollama/上游不可用 | 稍后重试；持续失败联系服务方 |

## 6. 容量与运维说明

- 单实例、单并发是**显存红线**下的设计，不是配置遗漏；`/v1` 的闸门与内部前端共享同一把推理锁；
- 服务端默认 `V1_QUEUE_TIMEOUT=30`（秒），可在 `.env` 调整——调大只会让调用方等更久，不会提高吞吐；
- 调用日志：问答本身不落调用方会话表（无状态），但服务端日志与监控看板仍按账号统计；
- 数据隔离：Key 绑定的系统账号与网页登录用户完全隔离，互相看不到知识库与历史。
