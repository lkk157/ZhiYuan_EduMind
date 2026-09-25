# -*- coding: utf-8 -*-
"""
长效记忆（M5）：错题直出事实 + 周报 LLM 提炼 + 双写（MySQL 事实源 / Chroma 语义索引）+ 注入召回。

三个刻意的设计取舍：
1. **不做「每问一提炼」**：每次提问已有 2–3 次串行 LLM 调用（改写/分类/生成），
   再加提炼会把首字延迟推到不可接受——提炼是低频重活，收在「生成周报」按钮里
   （用户明确要的时候才跑，且照旧走 gateway 串行闸门，显存红线自查通过）；
2. **错题型记忆零 LLM**：判分瞬间题干/对错/解析全是现成结构，模板拼句子即可；
3. **召回失败必须降级**：语义检索（embed+Chroma）任何一步挂掉 → 自动退回
   「最近 N 条」——个性化宁可变钝，绝不能把提问本身搞失败（索引是可抛弃的衍生组件）。

周报的报告正文（kind=report）**不参与注入**——它是给用户读的长文本，
塞进答疑 prompt 会挤爆 num_ctx；注入只取 weak_point/insight 且逐条截断。
"""
import json
import logging
import re

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.llm import gateway  # ★测试补丁缝：monkeypatch app.memory.long_term.gateway
from app.db import crud
from app.memory import knowledge_graph
from app.rag import embeddings
from app.rag.vector_store import memory_store_for

logger = logging.getLogger(__name__)

# 记忆类型（MemoryFact.kind 的合法值，禁止发明新值）
KIND_WEAK = "weak_point"
KIND_INSIGHT = "insight"
KIND_REPORT = "report"

# 注入与语义召回统一截断：单条背景最多 150 字（护 num_ctx，资料块才是主角）
_INJECT_CHARS = 150
# 周报素材上限：消息与错题都要截断——素材无界 = 改写式 prompt 爆炸
_REPORT_MESSAGES = 40
_REPORT_MESSAGE_CHARS = 200
_REPORT_RECORDS = 10
_REPORT_CONVERSATIONS = 5


def make_weak_fact(question: str, ref_file: str = "") -> str:
    """答错一题 → 一句结构化薄弱事实（零 LLM 的模板路径）。"""
    q = (question or "").strip().replace("\n", " ")[:60]
    where = f"（教材出处：{ref_file}）" if ref_file else ""
    return f"学生答错过：{q}{where}"


async def write_fact(
    db, *, user_id: int, kind: str, content: str, ref_file: str = ""
):
    """记忆双写：MySQL 先落（事实源），Chroma 索引 best-effort。

    为什么这个顺序：索引写失败只影响「语义召回的聪明程度」，
    而 MySQL 里没写等于记忆丢失——先保事实、再补索引；
    索引失败仅记日志（降级承诺），绝不让周报/判分因为索引挂了而报错。
    """
    fact = crud.create_memory_fact(
        db, user_id=user_id, kind=kind, content=content, ref_file=ref_file
    )
    try:
        await memory_store_for(user_id).upsert(
            ids=[f"mem{fact.id}"],
            texts=[content],
            metadatas=[{"user_id": user_id, "kind": kind}],
        )
    except Exception:
        logger.exception("记忆向量索引写入失败（降级：仅 MySQL），fact_id=%s", fact.id)
    return fact


async def recall_memories(db, *, user_id: int, question: str) -> list[str]:
    """答疑前的记忆召回：语义 top + 最近条目合并去重，截断到 memory_top_k 条。

    快路：一条记忆都没有 → 直接 []（全新用户的每次提问零额外开销）；
    语义段任何异常 → 静默降级最近条（注入是增强，不能反过来拖垮提问）。
    """
    limit = settings.memory_top_k
    recent = crud.list_memory_facts(
        db, user_id=user_id, limit=limit * 2, kinds=(KIND_WEAK, KIND_INSIGHT)
    )
    if not recent:
        return []  # 快路：没有记忆就没有后续任何模型/索引开销

    semantic: list[str] = []
    try:
        vectors = await embeddings.embed_texts([question])
        if vectors:
            for hit in memory_store_for(user_id).query(vectors[0], limit):
                text = (hit.text or "").strip()
                if text:
                    semantic.append(text)
    except Exception:
        logger.exception("记忆语义召回失败（降级：仅最近条）")

    # 合并：语义相关在前（与当前问题更贴），最近条补位；内容字符串去重
    merged: list[str] = []
    seen: set[str] = set()
    for text in semantic + [f.content for f in recent]:
        if text in seen:
            continue
        seen.add(text)
        merged.append(text[:_INJECT_CHARS])
        if len(merged) >= limit:
            break
    return merged


def parse_report(raw: str) -> dict | None:
    """周报 LLM 输出 → {"report": str, "weak_points": [str]}；不合规返回 None。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    report = obj.get("report")
    weak = obj.get("weak_points")
    if not isinstance(report, str) or not report.strip():
        return None
    if not isinstance(weak, list) or not all(isinstance(w, str) and w.strip() for w in weak):
        return None
    return {"report": report.strip(), "weak_points": [w.strip() for w in weak][:6]}


def collect_report_materials(db, *, user_id: int) -> str:
    """聚合周报素材：近期对话 + 答题记录（含章节锚与图谱关联，代码算好喂给 LLM）。

    素材分三段并全部设上限——素材无界会把 num_ctx 撑爆（红线）；
    图谱推荐数据由 knowledge_graph 现算（结构边零成本），LLM 只负责写成人话。
    """
    lines: list[str] = []

    # 1) 近期对话（最近 N 个会话的消息，逐条截断）
    convs = crud.list_conversations(db, user_id=user_id)[:_REPORT_CONVERSATIONS]
    msg_budget = _REPORT_MESSAGES
    for conv, _count in convs:
        lines.append(f"## 会话：{conv.title}")
        for msg in crud.list_messages(db, user_id=user_id, conversation_id=conv.id):
            if msg_budget <= 0:
                break
            msg_budget -= 1
            body = (msg.content or "").replace("\n", " ")[:_REPORT_MESSAGE_CHARS]
            who = "学生" if msg.role == "user" else "助手"
            lines.append(f"- {who}：{body}")
        if msg_budget <= 0:
            break

    # 2) 答题记录（含对错与教材锚）
    records = crud.list_quiz_records(
        db, user_id=user_id, limit=_REPORT_RECORDS, wrong_only=False
    )
    if records:
        lines.append("## 近期答题记录（对/错）")
        file_names = [d.file_name for d in _all_documents(db, user_id)]
        for r in records:
            mark = "对" if r.is_correct else "错"
            anchor = f"（{r.ref_file}）" if r.ref_file else ""
            lines.append(f"- [{mark}] {r.question[:80]} {anchor}")

    # 3) 薄弱章节的图谱关联（结构边，代码算好：周报里直接给「复习路径」）
    wrong_files = sorted({r.ref_file for r in records if r.ref_file and not r.is_correct})
    if wrong_files:
        lines.append("## 薄弱章节的关联关系（供复习路径建议）")
        group_ids = [g.id for g in crud.list_kb_groups(db, user_id=user_id)]
        for weak in wrong_files:
            related = knowledge_graph.related_for(user_id, weak, file_names, group_ids)
            if related:
                rel_txt = "；".join(f"{it.label}（{it.relation}）" for it in related)
                lines.append(f"- {weak} ← {rel_txt}")
    return "\n".join(lines)


def _all_documents(db, user_id: int):
    """用户全部分组下的文档（素材聚合用，跨组收集）。"""
    docs = []
    for group in crud.list_kb_groups(db, user_id=user_id):
        docs.extend(crud.list_documents(db, user_id=user_id, group_id=group.id))
    return docs


async def generate_report(db, *, user_id: int) -> dict:
    """生成学习周报（M5 唯一新增的 LLM 调用点，按钮触发、串行过闸门）。

    产出：报告正文（kind=report 入库，不注入）+ 薄弱点事实（kind=weak_point 入库且注入）。
    输出解析失败抛 RuntimeError → api 层翻译 502 人话（宁可报错不给假报告）。
    """
    from app.rag.prompts import build_report_prompt

    materials = collect_report_materials(db, user_id=user_id)
    if not materials.strip():
        raise AppError("最近没有可总结的对话与答题记录", code=400)

    system, prompt = build_report_prompt(materials)
    raw = await gateway.generate(
        model=settings.llm_model,
        prompt=prompt,
        system=system,
        num_predict=1024,  # 报告是本阶段最长输出，仍然有界（红线）
        temperature=0.3,
    )
    parsed = parse_report(raw)
    if parsed is None:
        logger.warning("周报 JSON 解析失败 raw=%.300s", raw)
        raise RuntimeError("学情报告解析失败")

    # 双写：报告本体 + 每条薄弱点（薄弱点进注入池，报告只做展示）
    report_fact = await write_fact(db, user_id=user_id, kind=KIND_REPORT, content=parsed["report"])
    weak_facts = []
    for point in parsed["weak_points"]:
        weak_facts.append(
            await write_fact(db, user_id=user_id, kind=KIND_WEAK, content=point)
        )
    return {
        "report": parsed["report"],
        "weak_points": parsed["weak_points"],
        "fact_ids": [report_fact.id] + [f.id for f in weak_facts],
    }
