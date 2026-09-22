# 知源 · 目录结构与开发路线图

> 本档配合 [README.md](../README.md)（技术栈与生产 vs 降级对比表）与 [CLAUDE.md](../CLAUDE.md)（协作规范）使用。

## 一、完整目录结构树

```
ZhiYuan_EduMind/
├── README.md                 # 项目总览：技术栈清单 + 生产vs降级对比表 + 快速开始
├── CLAUDE.md                 # 人机协作规范：角色/五步节奏/8G显存红线/git门禁（流程之锚）
├── requirements.txt          # 依赖清单（按里程碑分组注释，新手可知每个包何时引入）
├── .env.example              # 配置模板（真实 .env 含密码，绝不入库）
├── .gitignore                # 忽略 .env / data / logs / 缓存 / venv
├── docs/
│   └── ROADMAP.md            # 本文件：目录树 + M0–M8 计划
│
├── app/                      # ★ 后端主程序（FastAPI）
│   ├── main.py               #   应用入口：创建 app、挂路由、启动建表/初始化
│   ├── api/                  #   接口层：只做参数校验+调能力+返回，不写业务算法（薄）
│   │   ├── auth.py           #     注册/登录/JWT 签发与解析
│   │   ├── kb.py             #     知识库：分组 CRUD、上传、重建、删除
│   │   ├── chat.py           #     问答：对话、追问、历史
│   │   ├── quiz.py           #     出题/错题/题库
│   │   └── monitoring.py     #     统计数据给看板
│   ├── core/                 #   横切能力（被各层共用）
│   │   ├── config.py         #     pydantic-settings 读 .env，全局唯一配置源
│   │   ├── security.py       #     JWT 签发/校验、密码哈希
│   │   ├── llm.py            #     ★ Ollama 封装：推理互斥锁+Semaphore+keep_alive（显存红线所在）
│   │   └── exceptions.py     #     业务异常与错误码（前端好提示）
│   ├── db/                   #   数据访问层（MySQL = 事实源）
│   │   ├── session.py        #     SQLAlchemy engine/session（单测可切 SQLite 内存库）
│   │   ├── models.py         #     ORM：users/kb_groups/documents/chunk_fingerprints/
│   │   │                     #           conversations/messages/qa_logs/retrieval_logs/
│   │   │                     #           quiz_records/memory_facts
│   │   └── crud.py           #     增删改查封装（api 不直接写 SQL）
│   ├── ingest/               #   入库流水线（写路径：慢、可异步、耗显存）：解析→切块→指纹→向量化
│   │   ├── parsers/
│   │   │   ├── pdf_parser.py #     PDF 按页抽文本（无文本层页标记占位，留给 OCR）
│   │   │   ├── pptx_parser.py#     PPT 文本+图片切出，图片交 OCR（图表/公式）
│   │   │   ├── docx_parser.py#     Word 解析
│   │   │   └── image_parser.py#    单图 OCR 入库
│   │   ├── chunker.py        #     页感知切块（保留 file_name+page_no = 溯源根基）
│   │   ├── fingerprint.py    #     文件级+chunk 级 sha256，增量判断（不重建整库）
│   │   ├── ocr.py            #     视觉模型调用（图表/公式→文本），keep_alive=0
│   │   └── pipeline.py       #     编排整条入库流程（幂等、可重跑）
│   ├── rag/                  #   检索增强（读路径：快、必须稳）
│   │   ├── embeddings.py     #     调 Ollama embedding（批量限流+重试）
│   │   ├── vector_store.py   #     VectorStore 接口 + ChromaDB 实现（生产=Qdrant 实现）
│   │   ├── retriever.py      #     top-k 检索、分数阈值过滤、分组隔离
│   │   └── prompts.py        #     prompt 模板（仅依据资料）+ 来源强制拼接器
│   ├── agent/                #   智能调度（LangGraph 状态图）
│   │   ├── graph.py          #     意图节点→分发→工具节点→汇总
│   │   ├── intent.py         #     意图分类（强制枚举输出，非法默认=答疑）
│   │   └── tools.py          #     工具：知识库检索/计算器/生成题库/总结
│   ├── memory/               #   长短记忆
│   │   ├── short_term.py     #     滑动窗口（读 MySQL 最近 N 轮）
│   │   └── long_term.py      #     记忆提炼+语义召回（双写）
│   └── monitoring/           #   日志与统计
│       ├── logger.py         #     问答/检索日志写 MySQL + 文件日志
│       └── stats.py          #     聚合：提问量、高频知识点、命中率
│
├── frontend/                 # ★ Streamlit 界面（只做展示交互，业务全走 HTTP → 天然「多端」）
│   ├── app.py                #   主入口：登录/侧边栏导航/页面路由
│   ├── pages/                #   上传管理 / 智能答疑 / 题库错题 / 数据看板
│   ├── services/             #   调 FastAPI 的 HTTP 封装（带 JWT）
│   └── components/           #   来源卡片、对话气泡、指标卡等复用组件
│
├── tests/                    # ★ pytest 单测（每阶段必须全绿才能 push）
│   ├── conftest.py           #   三大 fixture：SQLite 内存库 / EphemeralChroma / Mock LLM
│   ├── test_smoke.py         #   骨架完整性（M0）
│   ├── test_security.py      #   JWT/密码（M1）
│   ├── test_chunker.py       #   切块与页码元数据（M2）
│   ├── test_fingerprint.py   #   增量指纹（M2）
│   ├── test_retriever.py     #   检索+兜底阈值（M3）
│   ├── test_citation.py      #   来源强制拼接（M3）
│   ├── test_intent.py        #   意图分类路由（M6）
│   └── test_memory.py        #   窗口/提炼/召回（M7）
│
├── scripts/                  #   运维/演示脚本（给人跑的）
│   ├── check_ollama.sh       #   模型拉取/连通性/显存检查（含 fallback 模型探测）
│   ├── init_db.sql           #   建库建表 SQL（幂等，utf8mb4）
│   ├── seed_demo.py          #   造演示数据
│   └── demo_e2e.py           #   端到端演示：上传→入库→提问→校验溯源（答辩自检）
│
├── config/                   #   非敏感静态配置（演示说明、阈值标定记录）
├── data/                     #   运行时数据（gitignore）：uploads/ 上传课件、chroma/ 向量库
└── logs/                     #   运行日志（gitignore）
```

**为什么这么分层（答辩可直接讲）：** ① api 薄、能力厚，改界面不碰坏检索；② ingest（写路径）与 rag（读路径）分离，显存互斥策略正好按这条线切；③ db 独立且 MySQL 是事实源、向量库是衍生索引，单测可换 SQLite；④ agent/memory/monitoring 可拔插——M3 窄条完全不依赖它们；⑤ frontend 只走 HTTP，体现多端架构。

---

## 二、开发节点规划 M0–M8（窄条优先）

> 总原则：**M3 出首个可演示闭环**（答辩信心保障），之后按「多轮→多模态→Agent→记忆→看板」扩展。
> 每阶段固定节奏（CLAUDE.md 强制）：**思路→用户确认→代码→单元测试→git push（打 tag）**。
> 需求覆盖映射：需求1多模态知识库→M2+M5｜需求2溯源RAG→M3+M4｜需求3Agent→M6｜需求4记忆→M4+M7｜需求5个性化→M7｜生产思维（并发显存/鉴权/增量/日志）→M1+M2+M5+M8。

| 里程碑 | 目标 | 关键产出 | 验收标准 | 主要风险 |
|---|---|---|---|---|
| **M0 项目骨架与协作规范**（0.5天） | 仓库、文档、规则、冒烟测试 | README、CLAUDE.md、ROADMAP、.gitignore/.env.example/requirements.txt、test_smoke.py | `pytest -q` 秒过；GitHub 可见；README 含 9 行对比表 | SSH/CRLF/中文乱码（见 README 快速开始排查命令） |
| **M1 基础设施打通**（2–3天） | 配置、MySQL 建表、JWT 注册登录、Ollama 连通+**推理互斥锁** | app/core/{config,security,llm,exceptions}.py、app/db/*、api/auth.py、scripts/{check_ollama.sh,init_db.sql} | JWT 往返/密码哈希单测过；curl 注册→登录拿 token；check_ollama.sh 打印模型清单+显存基线 | 模型未齐→check 脚本探测 fallback；MySQL 时区/编码→统一 utf8mb4 |
| **M2 文本入库链路**（4–5天） | PDF/Word 解析→页感知切块→两级指纹→embedding→Chroma 分组隔离（暂无 OCR） | app/ingest/{parsers,chunker,fingerprint,pipeline}、app/rag/{embeddings,vector_store}、api/kb.py | 切块含页码；重复上传跳过；改一块只重算一块；手动上传 PDF 后 MySQL+Chroma 可查 | 扫描版 PDF 无文本层→标记占位留给 M5 |
| **M3 ★窄条可演示闭环**（4–5天）**首个答辩演示点** | 界面跑通「登录→上传→入库→提问→强制【来源：文件名，页码】→无命中兜底」 | rag/{retriever,prompts}、api/chat.py、frontend/app.py+上传/答疑两页、demo_e2e.py | 单测：阈值兜底、来源拼接、非法引用剔除；demo_e2e 正例有来源+负例兜底不编造；tag `M3-thin-slice-demo` | 相似度阈值拍脑袋→3正例/3负例标定 |
| **M4 多轮追问+检索增强**（3天） | 滑窗短期记忆；模糊/术语/对比问法的 query 改写；兜底话术产品化 | memory/short_term.py、retriever 扩展 | 连续追问「它呢？那第二种呢？」指代正确 | 窗口挤爆 num_ctx→N=6+截断 |
| **M5 多模态入库（OCR）**（4–5天） | PPT/图片图表公式经视觉模型转文本入库；**显存互斥方案落地** | ingest/{ocr.py,parsers}、core/llm.py 模型切换锁 | 含公式 PPT 提问有来源；入库全程 nvidia-smi 无 OOM（记录峰值）；tag | **模型不可用**→fallback `qwen2.5-vl:3b`（见风险预案）；OCR 慢→异步入库+进度条 |
| **M6 Agent 智能调度**（5天） | 意图分类（答疑/出题/总结/计算）→LangGraph 分发工具；多轮可切换意图 | agent/{graph,intent,tools}.py | 同会话「出3道题→总结上一节→这道题怎么算」路由正确；非法标签默认答疑单测过 | 7B 分类不稳→温度0+枚举+few-shot |
| **M7 长效记忆+个性化**（4–5天） | 记忆提炼与语义召回（双写）；错题解析、知识点关联推荐、题库完善 | memory/long_term.py、api/quiz.py、models 扩表 | 新会话能「记得」用户弱点并调整讲解；user_id 隔离单测过 | 7B 提炼质量→模板固定字段；隐私→仅本用户可见 |
| **M8 监控看板+答辩收尾**（3–4天） | 日志完善、Streamlit 看板（提问量/高频知识点/命中率）、README 定稿、演示脚本 | monitoring/*、pages/dashboard.py、demo_e2e 完整版 | 跑 20 条问答看板对账；5 分钟全流程演示不 OOM | 看板慢→加时间范围/索引（体量小） |

> 节奏建议：每 3–7 天一个里程碑（纯开发约 5–7 周），完成后空一周回归+打磨演示；在 M3/M5/M6 各安排一次预答辩彩排。每阶段完成后按规范：**单测全绿 → `git push`**，里程碑额外打 tag。

---

## 三、风险与降级预案（要点）

| # | 风险 | 预案 |
|---|---|---|
| R1 | **DeepSeek-OCR 拉取失败或识别质量不佳** | M1 的 `check_ollama.sh` 探测 `deepseek-ocr` 可用性与识别效果；fallback 视觉模型 `qwen2.5-vl:3b`；`.env` 的 `OCR_MODEL` 一键切换，代码不写死 |
| R2 | **8G 显存 OOM** | 互斥锁 + Semaphore(1)；OCR `keep_alive=0` 且与 LLM 错峰；`num_ctx≤4096`；M5 验收必查 nvidia-smi 峰值；极端时 LLM 降 `qwen2.5:3b` 保底 |
| R3 | 7B 意图分类不稳 | 温度 0 + 强制枚举 + max_tokens≤16 + few-shot；非法默认「答疑」 |
| R4 | Windows 路径/编码/CRLF | pathlib + `encoding="utf-8"`；`core.autocrlf input` + `core.quotepath false` |
| R5 | 扫描版 PDF 无文本层 | M2 标记占位 → M5 OCR 接管 |
| R6 | 依赖版本漂移（chromadb/langgraph） | requirements 给下限版本；每里程碑用干净 venv 冒烟一次 |
