# ---------------------------------------------------------------------------
# 知源 ZhiYuan_EduMind 应用层镜像（M7-1）
#
# 设计边界（为什么不是「全部塞进容器」）：
# 1. Ollama 与模型留在宿主机：Windows + RTX 4060 的 GPU 透传依赖 WSL2 驱动与
#    NVIDIA Container Toolkit，答辩机不稳；更重要的是显存互斥锁是进程内实现，
#    模型服务与后端同机直连（host.docker.internal）才能保证「单实例单推理」可证。
# 2. MySQL 留在宿主机：演示机已有真实数据（课件/会话/错题），容器化要迁移数据
#    与卷权限，收益为零、风险为正。
# 3. 本镜像只含 Python 应用（FastAPI + Streamlit），data/logs 通过 compose 卷挂载。
#
# 多阶段构建：build-essential 只进 builder 层——若某个依赖没有预编译 wheel，
# 编译发生在构建期；运行镜像保持 slim，不背编译器与源码。
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# PIP_NO_CACHE_DIR：构建层不留 pip 缓存，镜像更小
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 部分科学计算依赖在无 wheel 时需要 gcc（有 wheel 时这些包不会被用到）
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依赖装进独立 venv：运行阶段整目录拷走，不重复解析依赖
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt


FROM python:3.11-slim AS runtime

# UTF-8：中文文件名/日志在 Linux 容器里必须显式声明（与 Windows 开发机同纪律）
# PYTHONUNBUFFERED：日志实时进 docker logs，不从缓冲区一股脑吐
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8

WORKDIR /app

# venv 从 builder 整体拷贝；应用代码按需拷入（.env 已由 .dockerignore 排除）
COPY --from=builder /opt/venv /opt/venv
COPY app/ app/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY config/ config/
COPY tests/ tests/
COPY requirements.txt .
# 项目契约文档也进镜像：test_smoke 锁定的就是这些交付物，
# 容器内自检（docker compose run --rm api pytest）必须与仓库同一份事实
COPY CLAUDE.md README.md .gitignore .env.example ./
COPY docs/ docs/

# 非 root 运行（容器逃逸/误写系统目录的纵深防御）；
# data/logs 预建并授权，compose 绑定挂载后由宿主机目录覆盖内容
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# 后端与前端共用本镜像，由 compose 的 command 决定角色
EXPOSE 8000 8501

# 默认跑后端；web 服务在 compose 里覆盖为 streamlit 命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
