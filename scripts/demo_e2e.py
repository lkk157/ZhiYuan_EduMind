# -*- coding: utf-8 -*-
"""
M2 端到端演示脚本：一份 docx 从入库到问答的全链路自检（python scripts/demo_e2e.py）。

为什么要做这个脚本而不是只靠单测：
单测把 gateway / 向量库都换成假件，证明的是「代码逻辑对」；
答辩现场要证明的是「真服务 + 真 Ollama + 真 Chroma 串起来还能对」——
本脚本走真实 HTTP 接口（/auth /kb /chat），只在最外层断言用户能看到的结果，
正是答辩演示脚本：注册 → 建组 → 传课件 → 正例提问（必须带来源）→ 负例提问（必须兜底不编造）。

为什么正例断言盯「来源：」、负例断言盯「未找到」：
这两条正是本项目的两条卖点——溯源可核对、空召回不调 LLM 防幻觉。
演示成败就看这两条断言，所以单独列在检查清单里。

用法（需先本地启动服务：uvicorn app.main:app --port 8000）：
    python scripts/demo_e2e.py
    API_BASE_URL=http://127.0.0.1:8000 python scripts/demo_e2e.py
退出码：全部 PASS 才是 0，任一 FAIL 即 1（便于 CI / 演示前体检直接看 $?）。
"""
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

import httpx
from docx import Document

# 服务地址默认本机 8000 端口；允许环境变量覆盖（演示机与开发机端口不同时不用改代码）
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# 超时为什么读超时放到 300 秒：上传要走 embedding、提问要走 LLM 生成，
# Ollama 首次加载模型可能超过 1 分钟——超时设短了会把「慢」误报成「挂」
TIMEOUT = httpx.Timeout(30.0, read=300.0)

# 演示课件里的唯一知识点：正例提问必须能命中它，
# 文案自拟且足够独特，避免检索撞上无关语料导致误判
KNOWLEDGE_TEXT = (
    "知源演示知识点：梯度下降的学习率过大时，损失函数会在最优点附近震荡甚至发散，"
    "导致模型无法收敛。此时应当适当调小学习率，或改用学习率衰减策略。"
)

# 正例：直指上述知识点；负例：与课件完全无关，用于验证空召回兜底（防幻觉）
QUESTION_POSITIVE = "梯度下降的学习率过大会有什么后果？"
QUESTION_NEGATIVE = "请问番茄炒蛋要放多少盐？"

# 检查清单文案刻意用纯 ASCII：GBK 控制台下中文/emoji 可能乱码甚至崩溃
# （CLAUDE.md Windows 规范），排查信息里的原始响应用 ensure_ascii=True 打印同理
CHECKS: list[tuple[str, bool]] = []


def report(label: str, ok: bool, raw=None) -> bool:
    """打印一行 PASS/FAIL 并记入清单；失败时附原始响应便于排查。

    为什么失败要打印收到的原始响应：演示现场/CI 里最费时间的是「不知道服务回了啥」，
    把 status/body 原样倒出来，一眼能分清是接口变了、鉴权挂了还是模型没命中。
    原始响应按 ensure_ascii 序列化，避免 GBK 控制台被非 ASCII 字符搞崩。
    """
    status_text = "PASS" if ok else "FAIL"
    print(f"  [{status_text}] {label}")
    if not ok and raw is not None:
        # ensure_ascii=True：中文变 \\uXXXX 转义，任何代码页的控制台都能安全打印
        body = json.dumps(raw, ensure_ascii=True, default=str)
        print(f"         raw response: {body}")
    CHECKS.append((label, ok))
    return ok


def build_demo_docx(target: Path) -> None:
    """在临时目录造一份含唯一知识点的 docx（python-docx 直接写二进制容器）。

    为什么用 docx 而不是 txt：演示要覆盖真实解析器链路（docx 解析 → 分块 → 向量），
    txt 不走 parse_document 契约；用 python-docx 生成保证文件格式合法、解析器必认识。
    """
    doc = Document()
    doc.add_paragraph(KNOWLEDGE_TEXT)
    doc.save(str(target))  # python-docx 保存的是 zip 二进制，必须落盘为 .docx 后缀


def main() -> int:
    """跑完整演示流程，返回进程退出码（全 PASS 才是 0）。

    为什么步骤做成线性而不是抽成一堆小函数：演示脚本的可读性优先于复用性，
    答辩时从上往下念一遍就是完整故事线；每步失败即打印原始响应并直接退出——
    后续步骤依赖前序产物（token / group_id），继续跑只会制造噪音 FAIL。
    """
    print(f"[M2 E2E DEMO] API_BASE_URL={API_BASE_URL}")
    print("[M2 E2E DEMO] checklist")

    # 时间戳后缀防重名：脚本可反复跑，不撞已有用户名（也不依赖清理旧数据）
    username = f"demo_e2e_{int(time.time() * 1000)}"
    # 随机密码：演示用户不复用任何固定口令，脚本里不留硬编码密码（CLAUDE.md §5）
    password = secrets.token_urlsafe(12)

    with httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT) as client:
        # ---- 1. 注册 ----
        resp = client.post("/auth/register", json={"username": username, "password": password})
        if not report("register random user (201)", resp.status_code == 201, resp.text):
            return finish()

        # ---- 2. 登录拿 token ----
        resp = client.post("/auth/login", json={"username": username, "password": password})
        token = resp.json().get("access_token") if resp.status_code == 200 else None
        if not report("login get token", resp.status_code == 200 and bool(token), resp.text):
            return finish()
        auth = {"Authorization": f"Bearer {token}"}

        # ---- 3. 建分组「演示课件」 ----
        resp = client.post("/kb/groups", json={"name": "演示课件"}, headers=auth)
        group_id = resp.json().get("id") if resp.status_code == 201 else None
        if not report("create kb group (201)", resp.status_code == 201 and group_id is not None, resp.text):
            return finish()

        # ---- 4. 造 docx 并上传 ----
        with tempfile.TemporaryDirectory(prefix="zhiyuan_demo_") as tmp:
            docx_path = Path(tmp) / "demo_knowledge.docx"
            build_demo_docx(docx_path)
            # multipart 上传：字段名 file 与 POST /kb/groups/{gid}/documents 的 UploadFile 形参一致
            resp = client.post(
                f"/kb/groups/{group_id}/documents",
                files={
                    "file": (
                        docx_path.name,
                        docx_path.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                headers=auth,
            )
            upload_ok = resp.status_code == 200
            upload_body = resp.json() if upload_ok else {}
            # 批量响应契约（2026-09-24 起）：{results:[...], succeeded, failed}
            first = (upload_body.get("results") or [{}])[0]
            # 入库必须成功且真有块（chunk_count>=1）：只判 200 会漏掉「解析出空文本」的静默失败
            upload_ok = upload_ok and first.get("ok") is True and first.get("chunk_count", 0) >= 1
            if not report("upload docx and ingest chunks", upload_ok, resp.text):
                return finish()

        # ---- 5. 正例提问：必须命中且带溯源 ----
        resp = client.post(
            "/chat/ask",
            json={"question": QUESTION_POSITIVE, "group_ids": [group_id]},
            headers=auth,
        )
        positive_ok = resp.status_code == 200
        positive_body = resp.json() if positive_ok else {}
        if not report("positive ask hit=true", positive_ok and positive_body.get("hit") is True, resp.text):
            return finish()
        # 「来源：」是 append_sources 强制追加的真溯源标记（模型自写的伪来源已被 sanitize 删掉）
        report(
            "positive ask answer has sources tag",
            "来源：" in positive_body.get("answer", ""),
            resp.text,
        )
        report(
            "positive ask sources list non-empty",
            bool(positive_body.get("sources")),
            resp.text,
        )

        # ---- 6. 负例提问：必须兜底（hit=false 且话术含「未找到」），验证不调 LLM 防幻觉 ----
        resp = client.post(
            "/chat/ask",
            json={"question": QUESTION_NEGATIVE, "group_ids": [group_id]},
            headers=auth,
        )
        negative_ok = resp.status_code == 200
        negative_body = resp.json() if negative_ok else {}
        if not report("negative ask hit=false", negative_ok and negative_body.get("hit") is False, resp.text):
            return finish()
        # 兜底话术含「未找到」语义（FALLBACK_MESSAGE 固定文案），命中了它才说明走的是防幻觉分支
        report(
            "negative ask answer is fallback wording",
            "未找到" in negative_body.get("answer", ""),
            resp.text,
        )

    return finish()


def finish() -> int:
    """汇总检查清单并给出退出码。

    为什么统一从这里出口：无论中途在哪一步 return finish()，
    清单打印格式与退出码语义都只有一份，不会出现「失败了但没打清单」的漏网情况。
    """
    failed = [label for label, ok in CHECKS if not ok]
    print(f"[M2 E2E DEMO] total={len(CHECKS)} failed={len(failed)}")
    if failed:
        print("[M2 E2E DEMO] RESULT: FAIL")
        return 1
    print("[M2 E2E DEMO] RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    # 独立脚本入口：退出码直接透传给 shell，演示前体检看 $? 即知成败
    sys.exit(main())
