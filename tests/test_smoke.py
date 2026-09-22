# -*- coding: utf-8 -*-
"""
M0 冒烟测试：锁住项目骨架完整性。

为什么要有这组测试：
M0 还没有任何业务代码，但它交付的是「结构契约」（README/CLAUDE.md/依赖清单/目录树）。
冒烟测试把这些契约固化下来，防止后续任何重构误删关键文件、悄悄改掉技术栈。
纯本地执行、零外部依赖（不连 MySQL、不打 Ollama），新手 `pytest -q` 秒级通过。
"""
from pathlib import Path

# 项目根目录：tests/ 的上一级（用 pathlib，避免 Windows 路径分隔符问题——CLAUDE.md 约定）
ROOT = Path(__file__).resolve().parents[1]


def test_project_structure():
    """关键文档必须存在：它们是 M0 的正式交付物。"""
    for rel in [
        "README.md",
        "CLAUDE.md",
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "docs/ROADMAP.md",
    ]:
        assert (ROOT / rel).exists(), f"缺少交付物：{rel}"


def test_env_example_keys():
    """.env.example 必须覆盖全部关键配置键——它是新环境的唯一配置向导。"""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in [
        "MYSQL_HOST",
        "MYSQL_PASSWORD",
        "JWT_SECRET",
        "OLLAMA_BASE_URL",
        "LLM_MODEL",
        "OCR_MODEL",
        "EMBED_MODEL",
        "LLM_KEEP_ALIVE",
        "OCR_KEEP_ALIVE",
        "SCORE_THRESHOLD",
    ]:
        assert key in text, f".env.example 缺少配置键：{key}"


def test_requirements_pins_core_stack():
    """依赖清单必须含锁定技术栈——防止静默换栈（CLAUDE.md 第 4 节）。"""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ["fastapi", "sqlalchemy", "pyjwt", "chromadb", "langgraph", "streamlit", "pytest"]:
        assert pkg in text, f"requirements.txt 缺少核心依赖：{pkg}"


def test_readme_has_production_matrix():
    """README 必须含「生产环境 vs 毕设降级」对比表——论文/答辩的核心叙事。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "生产环境" in text and "毕设降级" in text
    # 9 行对比表的关键关注点必须全部覆盖
    for topic in ["并发", "显存", "鉴权", "增量", "向量库", "日志"]:
        assert topic in text, f"README 对比表缺少关注点：{topic}"


def test_claudemd_has_workflow_and_vram_rules():
    """CLAUDE.md 必须含五步节奏与显存红线——协作契约的两个核心。"""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "思路" in text and "测试" in text and "git" in text.lower()
    assert "8G" in text or "8g" in text
    assert "keep_alive" in text or "互斥" in text
