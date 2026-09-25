# -*- coding: utf-8 -*-
"""
FastAPI 应用入口：创建 app、挂路由、启动初始化、健康检查。

启动命令（项目根目录）：
    uvicorn app.main:app --reload --port 8000
"""
import logging

from fastapi import Depends, FastAPI

from app.api.auth import get_current_user
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.kb import router as kb_router
from app.api.memory import router as memory_router
from app.api.quiz import router as quiz_router
from app.core.exceptions import install_error_handlers
from app.core.llm import gateway
from app.db.models import User
from app.db.session import init_db

# 日志配置：Windows 控制台默认 GBK，日志文案避免 emoji/特殊符号（风险 R4，M1 冒烟踩过）
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zhiyuan")

app = FastAPI(title="知源 ZhiYuan_EduMind", version="0.1.0")

# 统一错误出口：业务异常 → 人话 JSON（见 core/exceptions.py）
install_error_handlers(app)

# 挂鉴权路由：/auth/register、/auth/login
app.include_router(auth_router)
# 挂知识库路由：/kb/groups、/kb/documents（M2 接口层）
app.include_router(kb_router)
# 挂问答路由：/chat/ask（M2 接口层）
app.include_router(chat_router)
# 挂错题与记忆路由（M5）：/quiz/records、/memory/report 等
app.include_router(quiz_router)
app.include_router(memory_router)


@app.on_event("startup")
def _startup() -> None:
    """启动时幂等建表（M1 用 create_all 足够；生产演进方向是 Alembic 迁移）。"""
    init_db()
    logger.info("数据库表已就绪")


@app.get("/health")
async def health(user: User = Depends(get_current_user)) -> dict:
    """健康检查（需登录）：一次看清「鉴权 OK + Ollama 连通 + 三模型在列」。

    为什么需要登录：顺便验证 JWT 链路——演示/部署时注册→登录→打这个接口，
    三步就能确认整条鉴权+环境链路（与 scripts/check_ollama.sh 的排查思路一致，这里是 HTTP 版）。

    返回示例：
    {"status":"ok","user_id":1,"username":"demo",
     "ollama":{"ok":true,"models":[...],"missing":[]}}
    """
    return {
        "status": "ok",
        "user_id": user.id,
        "username": user.username,
        "ollama": await gateway.health(),
    }
