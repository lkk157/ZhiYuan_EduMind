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
│   └── ROADMAP.md            # 本文件：目录树 + M0–M6 计划（2026-09-23 重排）
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
│   ├── test_parsers.py       #   PDF/Word/PPT 解析与无文本层占位（M2）
│   ├── test_chunker.py       #   切块与页码元数据（M2）
│   ├── test_fingerprint.py   #   增量指纹（M2）
│   ├── test_retriever.py     #   检索+兜底阈值+分组隔离（M2）
│   ├── test_citation.py      #   来源强制拼接（M2）
│   ├── test_kb_chat.py       #   知识库/问答接口（M2）
│   ├── test_intent.py        #   意图分类路由（M4）
│   ├── test_ocr.py           #   OCR 回填与显存互斥序列（M3）
│   └── test_memory.py        #   窗口/提炼/召回（M5）
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

**为什么这么分层（答辩可直接讲）：** ① api 薄、能力厚，改界面不碰坏检索；② ingest（写路径）与 rag（读路径）分离，显存互斥策略正好按这条线切；③ db 独立且 MySQL 是事实源、向量库是衍生索引，单测可换 SQLite；④ agent/memory/monitoring 可拔插——M2 窄条完全不依赖它们；⑤ frontend 只走 HTTP，体现多端架构。

---

## 二、开发节点规划 M0–M6（窄条优先）

> **2026-09-23 产品经理调整（计划变更记录）**：RAG 知识库与前端优先——原 M2+M3 合并为新 M2（阶段完成即可在前端登录、查看功能）；Agent 提前至新 M3 并吸收原 M4 多轮追问（需求③本就把「多轮追问」列在 Agent 名下，指代消解与模糊/对比提问改写是同一个 query 改写模块）；OCR/记忆/看板顺延重编号为 M4–M6。协作流程（五步节奏/显存红线/git 门禁/过程报告）不变。
> **2026-09-24 产品经理再调整**：多模态入库（OCR）提前为新 M3——RAG 部分优先做完整（扫描页/图片识别属知识库本体），Agent 顺延至新 M4；M3 完成后产品经理先亲自体验运行，再开 Agent。
> **2026-09-24 产品经理再调整（其二）**：Agent 继续延后；RAG 完善阶段追加「会话历史持久化」（conversations/messages 两表落 MySQL，问答可切换回看）——M4 多轮滑窗将直接复用这两张表，属为后续铺路的 RAG 本体功能，不是新里程碑。
> **2026-09-24 产品经理选定后续四方向（排序定稿）**：体验验收+SCORE_THRESHOLD 标定（进行中，无代码）→ **M4 Agent**（吸收「随堂测」=出题工具场景化、「引导式答疑/苏格拉底模式」=答疑 prompt 开关）→ **RAG 质量包**（BM25+向量混合检索 RRF、可选 CPU rerank、RAG 评估指标实验表）→ **体验增强包**（流式输出、溯源点开看原页、会话导出 Markdown、示例问题）→ **M5**（吸收「学习周报」=记忆数据展示层）→ **M6**。明确不做：前端换 Vue（改写为 README「前端可替换」架构叙事）、Docker/Redis、移动端 APP。各包开工前仍逐包走五步节奏。
> 总原则：**M2 出首个可演示闭环**（答辩信心保障），之后按「Agent→多模态→记忆→看板」扩展。
> 每阶段固定节奏（CLAUDE.md 强制）：**思路→用户确认→代码→单元测试→git push（打 tag）**。
> **2026-09-25 M5 完成记录**：吸收「学习周报」与「知识点图谱关联推荐」（后者按产品经理要求做成**动态派生图**：不存边表、上传即入图、语义边用存量向量零模型调用）。
> 需求覆盖映射：需求1多模态知识库→M2+M3｜需求2溯源RAG→M2+M4｜需求3Agent→M4｜需求4记忆→M4+M5｜需求5个性化→M5｜生产思维（并发显存/鉴权/增量/日志）→M1+M2+M3+M6｜教学特色（随堂测/引导式答疑/学习周报）→M4+M5吸收｜RAG质量包与体验增强包→M4 后插入（见 2026-09-24 排序定稿）。

| 里程碑 | 目标 | 关键产出 | 验收标准 | 主要风险 |
|---|---|---|---|---|
| **M0 项目骨架与协作规范**（0.5天）✅ | 仓库、文档、规则、冒烟测试 | README、CLAUDE.md、ROADMAP、.gitignore/.env.example/requirements.txt、test_smoke.py | `pytest -q` 秒过；GitHub 可见；README 含 9 行对比表 | SSH/CRLF/中文乱码（见 README 快速开始排查命令） |
| **M1 基础设施打通**（2–3天）✅ | 配置、MySQL 建表、JWT 注册登录、Ollama 连通+**推理互斥锁** | app/core/{config,security,llm,exceptions}.py、app/db/*、api/auth.py、scripts/{check_ollama.sh,init_db.sql} | JWT 往返/密码哈希单测过；curl 注册→登录拿 token；check_ollama.sh 打印模型清单+显存基线 | 模型未齐→check 脚本探测 fallback；MySQL 时区/编码→统一 utf8mb4 |
| **M2 ★RAG 知识库与前端闭环**（5–7天）**首个答辩演示点**（原 M2+M3 合并） | 文本入库链路 + 精准溯源问答 + 前端登录与功能页 | ingest/{parsers,chunker,fingerprint,pipeline}、rag/{embeddings,vector_store,retriever,prompts}、api/{kb,chat}.py、frontend（登录/上传管理/智能答疑）、demo_e2e.py | 切块含页码；重传同文件短路、改一块只重算一块；前端登录→上传→提问→强制【来源：文件名，第X页】；无关问题兜底不编造；tag `M2-rag-kb-demo` | 相似度阈值拍脑袋→3正例/3负例标定；扫描版 PDF 无文本层→标记占位留给 M4 |
| **M3 多模态入库（OCR）**（4–5天） | 扫描页/图片/图表公式经 GLM-OCR 转文本入库；**显存互斥方案落地**；上传即同步识别 | ingest/{ocr.py,parsers/image_parser.py}、pipeline 集成、llm.active_models、前端图片上传 | empty_pages 按清单回填成可检索块；图片文件可入库可提问带来源；入库全程 nvidia-smi 无 OOM（记录峰值）；tag `M3-ocr-multimodal` | **模型不可用**→fallback `deepseek-ocr`/`qwen2.5-vl:3b`（见风险预案）；PDF 渲染依赖 pymupdf（纯 pip）；OCR 失败页留 empty_pages 可重传 |
| **M4 Agent 智能调度与多轮增强**（5–6天）（原 M6+M4 合并）✅ | 意图分类（答疑/出题/总结/计算）→LangGraph 分发工具；多轮滑窗 + query 改写（模糊/术语/对比/指代）；**吸收教学特色**：随堂测=出题工具、引导式答疑=答疑 prompt 开关；新增 /chat/score 判分 | agent/{graph,intent,tools}.py、memory/short_term.py、components/quiz.py | 同会话「出3道题→总结上一节→这道题怎么算」路由正确；连续追问「它呢？」指代正确；非法标签默认答疑单测过；**120 passed；tag `M4-agent`** | 7B 分类不稳→温度0+枚举+few-shot；窗口挤爆 num_ctx→N=6+截断；检索前置保「空召回零调用」硬闸门（过程报告有载） |
| **M5 长效记忆+个性化**（4–5天）✅ | 记忆提炼与语义召回（双写）；错题解析、知识点关联推荐、题库完善 | memory/long_term.py、api/quiz.py、models 扩表 | 新会话能「记得」用户弱点并调整讲解；user_id 隔离单测过；**185 passed；动态图谱/周报/错题本落地；tag `M5-memory`** | 7B 提炼质量→模板固定字段；隐私→仅本用户可见 |
| **M6 监控看板+答辩收尾**（3–4天） | 日志完善、Streamlit 看板（提问量/高频知识点/命中率）、README 定稿、演示脚本 | monitoring/*、pages/dashboard.py、demo_e2e 完整版 | 跑 20 条问答看板对账；5 分钟全流程演示不 OOM | 看板慢→加时间范围/索引（体量小） |

> 节奏建议：每 3–7 天一个里程碑（纯开发约 5–7 周），完成后空一周回归+打磨演示；在 M2/M3/M4 各安排一次预答辩彩排。每阶段完成后按规范：**单测全绿 → `git push`**，里程碑额外打 tag；过程报告随阶段入库（CLAUDE.md §8）。

---

## 三、风险与降级预案（要点）

| # | 风险 | 预案 |
|---|---|---|
| R1 | **GLM-OCR 拉取失败或识别质量不佳** | M1 的 `check_ollama.sh` 探测 `glm-ocr` 可用性与识别效果；fallback：`deepseek-ocr`（6.7GB）→ `qwen2.5-vl:3b`；`.env` 的 `OCR_MODEL` 一键切换，代码不写死 |
| R2 | **8G 显存 OOM** | 互斥锁 + Semaphore(1)；OCR `keep_alive=0` 且与 LLM 错峰；`num_ctx≤4096`；M3 验收必查 nvidia-smi 峰值；极端时 LLM 降 `qwen2.5:3b` 保底 |
| R3 | 7B 意图分类不稳 | 温度 0 + 强制枚举 + max_tokens≤16 + few-shot；非法默认「答疑」 |
| R4 | Windows 路径/编码/CRLF | pathlib + `encoding="utf-8"`；`core.autocrlf input` + `core.quotepath false` |
| R5 | 扫描版 PDF 无文本层 | M2 标记占位 → M3 OCR 接管 |
| R6 | 依赖版本漂移（chromadb/langgraph） | requirements 给下限版本；每里程碑用干净 venv 冒烟一次 |
