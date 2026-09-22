# -*- coding: utf-8 -*-
"""
全局配置中心：从 .env / 环境变量加载全部配置（全局唯一配置源）。

为什么集中在这里（CLAUDE.md §5）：
1. 禁止硬编码密码、模型名、阈值——所有可变的东西都进 .env，改配置不改代码；
2. 模型名必须与 `ollama list` 输出的 NAME 一致（在 .env 里维护，代码只读取），
   这样换模型（比如 OCR fallback）只需要改一行配置；
3. 提供 DATABASE_URL 覆盖机制：单测/临时演示可以不碰 MySQL（用 SQLite），
   不用修改 .env 本身——这也是「生产 MySQL / 毕设可降级」的一个小体现。
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：app/core/config.py 的上两级，供相对路径解析用
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """读取 .env 的强类型配置对象（pydantic 会做类型转换与校验）。"""

    # case_sensitive=False：.env 里写 MYSQL_HOST 也能映射到 mysql_host 字段
    # extra="ignore"：环境里无关的变量不会让加载报错
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 数据库（MySQL = 事实源）=====
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "change_me"
    mysql_db: str = "zhiyuan"
    # 可选覆盖：设了它就用它（单测用 sqlite:///:memory:，临时演示用 sqlite 文件）
    database_url: str | None = None

    # ===== JWT 鉴权 =====
    jwt_secret: str = "change_me_random_32chars"
    jwt_expire_minutes: int = 1440  # 令牌有效期（分钟），默认 24 小时

    # ===== Ollama（模型名以 ollama list 的 NAME 为准）=====
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ocr_model: str = "glm-ocr"
    embed_model: str = "qwen3-embedding:0.6b"
    # 显存红线（CLAUDE.md §3）：LLM 热点驻留 / OCR 用完即卸
    llm_keep_alive: str = "10m"
    ocr_keep_alive: str = "0"
    llm_num_ctx: int = 4096  # 上下文上限，防 8G 显存溢出

    # ===== RAG 参数（阈值在 M3 用正负例标定后回填）=====
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    score_threshold: float = 0.35

    # ===== 路径（相对项目根目录）=====
    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    log_dir: Path = Path("logs")

    def get_database_url(self) -> str:
        """计算数据库连接串：DATABASE_URL 覆盖优先，否则按 MySQL 各字段拼接。

        为什么拼 utf8mb4：MySQL 8 默认字符集若是 utf8mb3，中文会变问号/乱码
        （Windows GBK 坑的亲戚，见 ROADMAP 风险 R4），连接串显式声明最稳。
        """
        if self.database_url:
            return self.database_url
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


# 进程级单例：所有模块统一 `from app.core.config import settings`
settings = Settings()
