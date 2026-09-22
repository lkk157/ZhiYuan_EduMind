# 知源 ZhiYuan_EduMind

基于**多模态 RAG + Agent** 的高校学科教育智能答疑系统：支持 PPT/PDF/Word/图片多模态知识库（图表/公式经视觉模型识别入库）、精准溯源问答（回答末尾强制【来源：文件名，页码】）、Agent 智能调度（答疑/出题/总结）、长效记忆与个性化辅助。
本科毕业设计项目，演示环境为单台 RTX 4060 笔记本；架构按生产思维设计，并给出 8G 显存约束下的降级落地方案。

## 运行环境
- Windows 11 · RTX 4060（8G 显存）/ 16G 内存（答辩演示同机）
- Python 3.10+ · MySQL 8.x · Ollama（**无需 Docker**）

## 技术栈清单
| 层次 | 选型 | 用途 |
|---|---|---|
| 大语言模型 | Ollama + Qwen2.5:7b（本机标签 `qwen2.5:7b-instruct-q4_K_M`） | 答疑、出题、总结、意图分类 |
| 视觉/OCR | Ollama + DeepSeek-OCR * | PPT/图片中图表、公式识别转文本 |
| 文本向量化 | Ollama + Qwen3-Embedding:0.6b | 文档切片与记忆片段嵌入 |
| 向量库 | ChromaDB（嵌入式持久化） | 切片/记忆语义检索；生产升级 Qdrant（见下表） |
| 数据库 | MySQL 8.x | 事实源：用户/会话/文档元数据/增量指纹/日志 |
| 后端 | FastAPI | API、入库流水线、RAG、Agent 编排、并发闸门 |
| 前端 | Streamlit | 演示界面（登录/知识库/答疑/题库/看板） |
| Agent 编排 | LangGraph | 显式意图分类 + 代码级工具路由（防 7B 工具调用漂移） |
| 鉴权 | JWT（PyJWT + bcrypt） | 多端登录，核心数据表绑定 user_id |
| 测试 | pytest / pytest-asyncio | 每阶段单元测试门禁 |

\* OCR 模型名以 `ollama pull` 实际结果为准；不可用时按「快速开始」中的 fallback 切换，仅需修改 `.env`。**所有模型名以 `ollama list` 输出的 NAME 为准。**

## 生产环境 vs 毕设降级方案（核心）
| # | 关注点 | 生产环境方案 | 毕设降级方案（本机落地） | 降级影响 | 升级路径 |
|---|---|---|---|---|---|
| 1 | 并发与显存治理 | Redis 分布式锁 + Celery 任务队列 + 独立 GPU 推理服务 | FastAPI 进程内 `asyncio.Semaphore` 排队 + 全局推理互斥锁 + `keep_alive` 控制 | 多用户并发时排队变慢（单人演示无感） | 本地锁换 Redis 分布式锁，推理改 RPC 到推理服务 |
| 2 | 推理服务 | vLLM/TGI 推理集群（OpenAI 兼容 API，模型常驻） | 本机 Ollama 单实例，`num_ctx` 限制、推理串行化 | 吞吐低、单点 | Ollama → vLLM（同为 OpenAI 接口，换 base_url） |
| 3 | OCR 多模态 | 独立视觉推理服务 + 异步 OCR 队列 + 结果缓存 | Ollama 拉起视觉模型，`keep_alive=0` 用完即卸，与 LLM 错峰互斥 | OCR 吞吐低、首次加载慢 | OCR 服务化 + GPU 独占 |
| 4 | 多端鉴权与同步 | OAuth2/SSO、刷新令牌、多端吊销、数据同步服务 | JWT 鉴权 + 核心表绑定 `user_id`（多端登录数据一致） | 无刷新吊销、无冲突合并 | refresh_token + Redis 黑名单 + 同步协议 |
| 5 | 数据一致性与增量更新 | CDC/消息队列驱动增量索引，chunk 版本化，灰度重建 | 文件级+chunk 级 sha256 指纹表（MySQL 事实源），只向量化变更块，不重建整库 | 单写者、无并发写冲突处理 | 消息队列 + 向量库版本快照 |
| 6 | 向量库 | Qdrant 集群（分片/副本/备份） | ChromaDB 嵌入式单机持久化（本地目录） | 单机容量与可用性 | `VectorStore` 接口换 Qdrant 实现类 |
| 7 | 日志与监控 | ELK/Loki + Prometheus + Grafana + 告警 | 问答/检索日志落 MySQL + Streamlit 数据看板 | 无告警；日志量大后查询慢 | 日志外接 Loki，看板换 Grafana |
| 8 | 任务调度 | K8s + 独立 worker 池 | FastAPI BackgroundTasks / 线程池执行入库 | 无重试/死信 | Celery worker + 重试策略 |
| 9 | 数据库 | MySQL 主从 + 读写分离 + 定期备份 | 本机 MySQL 8.x 单实例 | 单点、无自动备份 | binlog 主从 + 定时 mysqldump |

## 快速开始
1. **模型准备**（Ollama 已安装的前提下）：
   ```bash
   ollama pull qwen2.5:7b-instruct-q4_K_M   # LLM（本机已装则跳过）
   ollama pull qwen3-embedding:0.6b
   ollama pull deepseek-ocr                  # 视觉/OCR 定为 DeepSeek-OCR（若失败 fallback qwen2.5-vl:3b，
                                             # 并同步修改 .env 中 OCR_MODEL——代码不写死模型名）
   ollama list                               # 确认模型在列，并将 NAME 原样填入 .env
   ```
2. **配置**：`cp .env.example .env`，修改 `MYSQL_PASSWORD`（本机真实密码）与 `JWT_SECRET`（随机串）。
3. **安装依赖**：`pip install -r requirements.txt`
4. **冒烟测试**：`pytest -q`（应全部通过）
5. **Git 首次配置（Windows 防坑）**：
   ```bash
   git config core.autocrlf input      # 统一换行，避免 CRLF 警告
   git config core.quotepath false     # 中文文件名不乱码
   ssh -T git@github.com               # 验证 SSH 密钥（应返回 Hi lkk157!）
   ```
6. 启动命令将在 M1 起补充（`uvicorn` + `streamlit run`）。

## 目录结构与开发里程碑
完整目录树（含每文件职责标注）与 M0–M8 开发计划见 **[docs/ROADMAP.md](docs/ROADMAP.md)**。

## 协作与提交规范（摘要，详见 [CLAUDE.md](CLAUDE.md)）
- 节奏五步：**思路 → 确认 → 代码（详细中文注释讲“为什么”）→ 测试 → git push**，禁止一口气生成所有代码；
- 每阶段单元测试全绿后 push 到本仓库，里程碑打 tag；
- 8G 显存红线：大模型互斥驻留、OCR 用完即卸、推理串行化。
