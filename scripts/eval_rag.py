# -*- coding: utf-8 -*-
"""
RAG 评估与阈值标定脚本（RAG 质量包）：跑真实服务产出三张表 + 阈值建议。

为什么走 HTTP 而不是进程内直连：评估必须测「用户真正走的那条路」——
鉴权、检索、生成、兜底闸门全链路；进程内直连会绕过接口层，
测出「代码对」但用户拿到的仍是另一套行为（与 demo_e2e 同一哲学）。

产出（写入 docs/reports/rag-eval-<label>.md 并打印到控制台）：
1. 逐问题结果表：意图/是否命中/相似度分数/是否带来源；
2. 分数分布：正例最小分、负例（擦边命中时）最高分；
3. SCORE_THRESHOLD 建议值 = (正例最小分 + 负例最高分) / 2，由产品经理拍板后回填 .env。

用法（需先启动 uvicorn + Ollama，且已上传课件）：
    python scripts/eval_rag.py                      # 默认 label=run1
    python scripts/eval_rag.py --label hybrid       # 切 RETRIEVAL_MODE=hybrid 重启后跑
    python scripts/eval_rag.py --dataset my.json    # 自备问题集
退出码：全部正例命中且全部负例兜底 → 0，否则 1。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(30.0, read=300.0)  # 含 LLM 生成，读超时放宽（同 demo_e2e）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "config" / "eval_dataset.json"
REPORT_DIR = PROJECT_ROOT / "docs" / "reports"

# 与后端 FALLBACK_MESSAGE 一致的判定子串（负例必须落到这里才算防幻觉闸门生效）
FALLBACK_MARK = "未找到相关内容"


def load_dataset(path: Path) -> list[dict]:
    """读问题集 JSON：[{question, expect_hit, note?}]。文件不存在时给出人话指引。"""
    if not path.is_file():
        print(f"[eval] 找不到问题集: {path}")
        print("[eval] 请复制 config/eval_dataset.json 模板并按你的课件改写问题。")
        sys.exit(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if "question" not in item or "expect_hit" not in item:
            print(f"[eval] 问题集条目缺 question/expect_hit 字段: {item}")
            sys.exit(2)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评估与阈值标定")
    parser.add_argument("--label", default="run1", help="本轮标签（如 vector / hybrid）")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="问题集 JSON 路径")
    parser.add_argument("--username", default=os.environ.get("EVAL_USER", "demo"))
    parser.add_argument("--password", default=os.environ.get("EVAL_PASS", ""))
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    password = args.password or os.environ.get("MYSQL_PASSWORD") or ""
    if not password:
        print("[eval] 请设置环境变量 EVAL_PASS（或 --password）提供登录密码。")
        return 2

    rows: list[dict] = []
    with httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT) as client:
        # ---- 登录 ----
        r = client.post("/auth/login", json={"username": args.username, "password": password})
        if r.status_code != 200:
            print(f"[eval] 登录失败（{r.status_code}）: {r.text[:200]}")
            return 2
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

        # ---- 逐问题评估 ----
        for item in dataset:
            question = item["question"]
            expect_hit = bool(item["expect_hit"])
            t0 = time.time()
            r = client.post("/chat/ask", json={"question": question}, headers=headers)
            elapsed = time.time() - t0
            if r.status_code != 200:
                rows.append({"q": question, "expect": expect_hit, "error": f"HTTP {r.status_code}"})
                continue
            body = r.json()
            scores = [s.get("score") for s in body.get("sources", []) if s.get("score") is not None]
            rows.append(
                {
                    "q": question,
                    "expect": expect_hit,
                    "hit": bool(body.get("hit")),
                    "intent": body.get("intent", "?"),
                    "best": max(scores) if scores else None,
                    "has_source": "来源：" in body.get("answer", ""),
                    "fallback": FALLBACK_MARK in body.get("answer", ""),
                    "elapsed": round(elapsed, 1),
                }
            )
            print(f"[eval] {elapsed:5.1f}s hit={body.get('hit')} {question[:30]}")

    # ---- 统计与建议 ----
    pos_hits = [r_ for r_ in rows if r_["expect"] and r_.get("hit")]
    pos_scores = [r_["best"] for r_ in pos_hits if r_.get("best") is not None]
    neg_wrong = [r_ for r_ in rows if not r_["expect"] and r_.get("hit")]  # 该兜底却命中
    neg_scores = [r_["best"] for r_ in neg_wrong if r_.get("best") is not None]

    all_pos_ok = all(r_.get("hit") for r_ in rows if r_["expect"]) and not any(
        "error" in r_ for r_ in rows
    )
    all_neg_ok = all(not r_.get("hit") for r_ in rows if not r_["expect"])

    suggestion = ""
    if pos_scores and neg_scores:
        mid = (min(pos_scores) + max(neg_scores)) / 2
        suggestion = (
            f"SCORE_THRESHOLD 建议值 **{mid:.2f}**"
            f"（正例最低 {min(pos_scores):.2f} / 负例擦边最高 {max(neg_scores):.2f}）"
        )
    elif pos_scores:
        suggestion = (
            f"未见负例擦边命中：可取正例最低分 {min(pos_scores):.2f} 略降一档（如 {min(pos_scores) - 0.05:.2f}）"
            "作为 SCORE_THRESHOLD，保留余量。"
        )
    else:
        suggestion = "正例全部未命中——先检查课件是否入库、问题是否贴课件内容。"

    # ---- 报告落盘（markdown，直接进论文实验章节）----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"rag-eval-{args.label}.md"
    lines = [
        f"# RAG 评估报告 · {args.label}",
        "",
        f"> 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}｜问题集：{args.dataset}",
        "",
        "| 问题 | 期望 | 实际命中 | 意图 | 最高相似度 | 带来源 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r_ in rows:
        if "error" in r_:
            lines.append(f"| {r_['q']} | {'命中' if r_['expect'] else '兜底'} | ERROR:{r_['error']} | - | - | - | - |")
        else:
            lines.append(
                f"| {r_['q']} | {'命中' if r_['expect'] else '兜底'} | {'是' if r_['hit'] else '否'} "
                f"| {r_['intent']} | {r_['best'] if r_['best'] is not None else '-'} "
                f"| {'是' if r_['has_source'] else '否'} | {r_['elapsed']} |"
            )
    lines += ["", f"**阈值建议**：{suggestion}", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== 评估汇总 =====")
    print(f"正例全命中: {all_pos_ok}｜负例全兜底: {all_neg_ok}")
    print(f"阈值建议: {suggestion}")
    print(f"报告已写入: {report_path}")
    return 0 if (all_pos_ok and all_neg_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
