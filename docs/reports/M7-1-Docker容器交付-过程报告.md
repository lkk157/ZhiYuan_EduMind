# M7-1 过程报告 · 应用层 Docker 容器交付

> 所属阶段：M7-1（服务化里程碑上半程）｜完成日期：2026-10-20｜单测：192 passed

---

## 一、这阶段做了什么

1. **契约更新**（产品经理 2026-10-20 书面确认）：
   - CLAUDE.md §7 的 Docker 禁令改为「Docker 仅作交付形态」；
   - §4 技术栈补 Docker（应用层容器化，Ollama/MySQL 留宿主机）；
   - §3 红线补「容器化只允许后端单实例」（显存互斥锁是进程内实现）；
   - ROADMAP 新增 M7 行与计划变更记录。
2. **镜像** `Dockerfile`：python:3.11-slim 多阶段（builder 装 build-essential + venv，runtime 只拷 venv 与代码）；非 root 用户、UTF-8 环境、data/logs 预建；`CMD` 默认跑 uvicorn，web 角色由 compose 覆盖 command。
3. **编排** `docker-compose.yml`：api + web 两服务同镜像；`.env` 经 `env_file` 注入；容器内 `MYSQL_HOST`/`OLLAMA_BASE_URL` 覆盖为 `host.docker.internal`；data/logs 绑定挂载；api 健康检查基于新端点 `/healthz`；端口可用 `API_PORT`/`WEB_PORT` 覆盖（避免与原生服务冲突）；web `depends_on: api healthy`。
4. **`.dockerignore`**：排除 `.env`（密钥绝不进镜像层）、data/logs、缓存、venv、.git。
5. **新端点** `GET /healthz`：免鉴权、不碰 DB/Ollama——编排探针每 10 秒调用，不该持有 JWT，也不该把模型冷加载误判为容器故障。
6. **README**：技术栈补交付行；快速开始新增 Docker 运行章节，明确三条前置（MySQL 专用账号、Ollama 监听 0.0.0.0、端口占用）。
7. **环境适配（本机一次性）**：
   - MySQL 8.0.40 只有 `root@localhost`，容器来自 Docker 网关 IP 连不上 → 创建专用账号 `zhiyuan@'%'` 与 `zhiyuan@'localhost'`（只授权 `zhiyuan.*`），密码随机生成并回填 `.env`；
   - Docker 镜像源 `docker.xuanyuan.me` 403、Docker Hub 直连超时 → 测通 `docker.m.daocloud.io` 并置顶（原配置备份为 `daemon.json.bak-*`）；
   - requirements 显式补 `cryptography>=3.4`（PyMySQL 连接 MySQL 8 默认 `caching_sha2_password` 的必要依赖，容器干净环境无传递依赖）。

## 二、遇到了什么问题

1. 评估时 Docker 已安装但守护进程未运行（只有 compose 客户端）；
2. Docker 镜像源 `docker.xuanyuan.me` 免费节点 403/限流，`registry-1.docker.io` 直连超时，首次构建失败（compose 经管道吞了退出码，表面 exit 0）；
3. MySQL 账号只有 `root@localhost`——容器节点连接被拒；
4. 容器内 PyMySQL 缺 `cryptography` 时无法完成 MySQL 8 默认认证插件握手（开发机靠传递依赖侥幸可用）；
5. Ollama 未运行，全量单测出现 `ConnectError`——**tests/test_knowledge_graph.py 的 upsert 在打桩之前直连了真实 Ollama**：Ollama 在线时测试一直「侥幸全绿」，单测零网络纪律被无声破坏；
6. Ollama 监听模式不确定：默认只绑 `127.0.0.1`，且 Windows 下 Ollama 桌面端会自动拉起 serve；不同 Docker 引擎对 `host.docker.internal` 的回环转发行为不同，评估阶段未能给出确定结论。

## 三、怎么解决的问题（与上一一对应）

1. 先启动 Docker Desktop，`docker version` 确认 29.8.2 / linux-x86_64；
2. 备份 `~/.docker/daemon.json` 后把 `docker.m.daocloud.io` 置顶（实测 `/v2/` 返回 401=可达），`docker desktop restart` 生效；重新拉取 `python:3.11-slim` 成功；
3. 创建容器专用最小权限账号（`@'%'` + `@'localhost'`，仅授权目标库），不开放 root 远程；`.env` 回填新账号，原生与容器两条路径都可用；
4. requirements 显式声明 cryptography（注释写明原因）；
5. 修复测试：upsert 前先打 `fake_embed_fn` 假向量桩，数据写好后再换成「一调即失败」桩；修复后在 **Ollama 停机状态下全量 192 passed**——零网络纪律从「侥幸」变为「可证」；
6. README 前置条件改为「先自检容器 → Ollama 是否可达」：给出一行 `docker compose exec api` 探测命令；连不上时再设 `OLLAMA_HOST=0.0.0.0:11434` 并重启。本机实测 Windows Docker Desktop 的回环转发可直接访问 127.0.0.1 绑定的 Ollama（容器内 200），因此不强制改全局环境变量；同时发现 Ollama 桌面端会自行拉起 serve，排障时杀掉重复实例、确认单实例后才做问答（生成期间只有一个 Ollama 进程，红线成立）。

## 四、真机验证（2026-10-20，Docker 29.8.2 / linux-x86_64）

| 项 | 结果 |
|---|---|
| 镜像构建 | `zhiyuan-edumind:latest` 1.8GB，多阶段构建成功 |
| 容器状态 | api `Up (healthy)`、web `Up`；`/healthz` 200、Streamlit `:18501` 200 |
| 容器连 MySQL | 启动日志「数据库表已就绪」；注册/登录/知识库/看板接口全部 200 |
| 容器连 Ollama | `docker compose exec api` 内访问 `host.docker.internal:11434/api/tags` 200 |
| 绑定挂载 | 容器写入落在宿主机 `data/uploads`（双向可见，8 文件） |
| 容器内单测 | `docker compose run --rm api pytest -q` → **192 passed**（无 Ollama 依赖） |
| 端到端 | `API_BASE_URL=http://127.0.0.1:18000 python scripts/demo_e2e.py` → **17/17 ALL PASS**（含真 LLM 出题、判分、看板对账） |

> 验证时用 `API_PORT=18000 WEB_PORT=18501` 与原生服务（8000/8501）并行；正式使用默认端口即可。
> 复现命令：`docker compose up -d --build` → `docker compose ps` → 浏览器 `http://127.0.0.1:8501`。
