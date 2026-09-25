# -*- coding: utf-8 -*-
"""
Agent 工具层（M4）：答疑 / 出题 / 总结 / 计算 四个工具，LangGraph 各路由节点的执行体。

分层由来：问答主逻辑原本长在 api/chat.py 里——M4 引入路由后「接口层只做校验拼响应」
的约定要求算法下沉到能力层，本文件即它的新家。防幻觉三层防线（prompt 铁律 +
sanitize 删伪来源 + 空召回不调 LLM）、资料不足闸门、来源强制拼接全部**原样平移**，
语义一个字没动（平移前的单测断言继续成立）。

为什么所有工具共用本模块的 gateway/retrieve 命名空间：
单测的 monkeypatch 按「调用方命名空间」生效——缝收在一处（本文件），
测试只需补 app.agent.tools.* 一个点，四个工具同时被替换（与 store_for 四缝同一手法）。
"""
import ast
import json
import logging
import operator
import re
from typing import Sequence

from app.core.config import settings
from app.core.llm import gateway  # ★测试补丁缝：monkeypatch app.agent.tools.gateway
from app.rag.prompts import (
    FALLBACK_MESSAGE,
    append_sources,
    build_qa_prompt,
    build_quiz_prompt,
    build_score_prompt,
    build_summary_prompt,
    is_insufficient_answer,
    sanitize_answer,
)
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# 溯源摘要长度：snippet=块文本前约 80 字（契约口径，自 chat.py 平移）
SNIPPET_LEN = 80


def build_sources(chunks: Sequence["RetrievedChunk"]) -> list[dict]:
    """命中块 → 出口来源列表（(file_name, page_no) 去重保序，与 append_sources 同序同口径）。"""
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.file_name, chunk.page_no)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file_name": chunk.file_name,
                "page_no": chunk.page_no,
                "snippet": chunk.text[:SNIPPET_LEN],
                # 相似度分数透出（2026-09-25 质量优化）：用户与标定流程都看得见命中质量，
                # SCORE_THRESHOLD 的 3正3负标定从此有数据依据而不是拍脑袋
                "score": round(float(chunk.score), 4),
            }
        )
    return sources


def _fallback(intent: str) -> dict:
    """统一兜底口径：固定话术 + hit=false + 空来源（契约与 M2 完全一致）。"""
    return {"answer": FALLBACK_MESSAGE, "sources": [], "hit": False, "intent": intent}


async def _generate(
    *,
    system: str,
    prompt: str,
    num_predict: int,
    temperature: float,
    on_delta=None,
) -> str:
    """最终答案生成的双模入口：无回调=非流式（既有路径），有回调=流式逐段回吐。

    为什么收口在一个函数：qa/summary 两条流式路径与非流式路径的参数完全同源，
    分散写四处（两工具×两模式）迟早漂移；后处理（净化/闸门/拼来源）仍在各自工具里，
    本函数只管「怎么把字要回来」——流式只是传输，生成语义与红线（锁覆盖整段流，
    见 llm.generate_stream 注释）都不在这里重写。
    """
    kwargs = dict(
        model=settings.llm_model,
        prompt=prompt,
        system=system,
        num_predict=num_predict,
        temperature=temperature,
    )
    if on_delta is None:
        return await gateway.generate(**kwargs)
    pieces: list[str] = []
    async for piece in gateway.generate_stream(**kwargs):
        pieces.append(piece)
        on_delta(piece)  # 同步回调（SSE 场景=queue.put_nowait，绝不阻塞事件循环）
    return "".join(pieces)


# ===== 答疑工具（自 chat.py 原样平移）=====


async def qa_tool(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    guide_mode: bool = False,
    on_delta=None,
) -> dict:
    """命中资料后的标准问答：生成 → 净化伪来源 → 资料不足闸门 → 强制拼真来源。

    三层防线与闸门的语义见模块 docstring——本函数是它们唯一的现行宿主，
    改动前先想清楚「答辩演示的卖点还在不在」。
    on_delta：流式回调（体验增强包）——**闸门/净化在流完后统一做**，
    流出去的是原始 token、返回的 done 是净化结果，两边一致性由前端以 done 为准。
    """
    system, prompt = build_qa_prompt(question, chunks, guide=guide_mode)
    raw = await _generate(
        system=system,
        prompt=prompt,
        num_predict=512,  # 输出长度受控（显存红线）
        temperature=0.3,  # 低温度：问答忠于资料，减少发挥
        on_delta=on_delta,
    )
    answer = sanitize_answer(raw)
    # 资料不足闸门（2026-09-24）：模型说不知道 → 兜底、不拼来源
    if is_insufficient_answer(answer):
        return _fallback("qa")
    answer = append_sources(answer, chunks)
    return {"answer": answer, "sources": build_sources(chunks), "hit": True, "intent": "qa"}


# ===== 出题工具（含随堂测）=====


def parse_quiz(raw: str) -> dict | None:
    """把模型输出解析成合法试题 JSON；任何不合规输出返回 None（调用方走兜底）。

    为什么要剥代码围栏：即使命令里写了「不要输出 ```」，7B 仍可能手痒包一层——
    解析器做足防御比指望提示词百分百听话现实。
    """
    text = (raw or "").strip()
    # 防御性剥 ```json ... ``` 围栏
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    # ---- 结构校验（宁可兜底也不渲染半残卡片）----
    if not isinstance(obj, dict) or obj.get("type") != "quiz":
        return None
    questions = obj.get("questions")
    if not isinstance(questions, list) or not questions:
        return None
    for q in questions:
        if not isinstance(q, dict):
            return None
        if q.get("type") not in {"choice", "short"}:
            return None
        if not str(q.get("question", "")).strip():
            return None
        if not str(q.get("answer", "")).strip():
            return None
        if q["type"] == "choice":
            options = q.get("options")
            if not isinstance(options, list) or len(options) < 2:
                return None
            # ---- 多选识别与校验（2026-09-25 反馈：单选框容不下多选题）----
            # answer 允许一个或多个字母（"A"=单选 / "AB"=多选），但必须：
            # 纯字母、无重复、不超过选项数——不合法说明模型输出坏掉，整卷拒收走兜底
            ans = str(q["answer"]).strip().upper()
            if not ans.isalpha() or len(set(ans)) != len(ans):
                return None
            if any(ch > chr(ord("A") + len(options) - 1) for ch in ans):
                return None
            # 归一化为升序（"BA"→"AB"）：渲染与判分都按同一字面比较，不吃顺序差异
            q["answer"] = "".join(sorted(ans))
    return obj


async def quiz_tool(question: str, chunks: list[RetrievedChunk]) -> dict:
    """出题/随堂测：依据命中资料生成试题 JSON（answer 字段=JSON 字符串，不拼【来源】）。

    为什么不 append_sources 到 answer：answer 本体就是 JSON 文本，
    末尾拼一串【来源：…】会直接破坏 JSON 结构——溯源改由 sources 字段独立承载，
    前端试题卡与来源卡片并排展示，信息一点不少。
    JSON 解析失败走统一兜底口径（hit=false + 固定话术）：宁可让用户重问一次，
    也不返回半残 JSON 让前端渲染报错；失败细节进日志排查。

    top_k 截断：检索池按全工具最大值召回（graph 层），出题只要最聚焦的前
    top_k_quiz 块——材料越杂题目越散（2026-09-25 反馈的根因之一）。
    """
    chunks = chunks[: settings.top_k_quiz]
    system, prompt = build_quiz_prompt(question, chunks)
    raw = await gateway.generate(
        model=settings.llm_model,
        prompt=prompt,
        system=system,
        # 出题输出最长（5 题×JSON），上限 1024 token——仍是有界输出（红线），
        # 且 num_ctx=4096 只约束输入侧，1024 输出不会挤爆上下文
        num_predict=1024,
        temperature=0.4,  # 适度升温：题目措辞需要一点多样性，但结构由 JSON 约束兜住
    )
    quiz = parse_quiz(raw)
    if quiz is None:
        logger.warning("出题 JSON 解析失败，走兜底。raw=%.200s", raw)
        return _fallback("quiz")
    # 归一化重序列化：前端/落库拿到的都是规范化 JSON（键序、空白一致，解析稳定）
    answer = json.dumps(quiz, ensure_ascii=False)
    return {"answer": answer, "sources": build_sources(chunks), "hit": True, "intent": "quiz"}


# ===== 总结工具 =====


async def summary_tool(
    question: str, chunks: list[RetrievedChunk], *, on_delta=None
) -> dict:
    """总结：依据命中资料输出要点列表（净化/闸门/拼来源与问答完全同构）。

    top_k_summary 默认比问答池大：总结要覆盖更全，配合范围过滤（页码/章节）
    「广而有界」——范围由 scope 管，条数由本工具管，两层各司其职。
    """
    chunks = chunks[: settings.top_k_summary]
    system, prompt = build_summary_prompt(question, chunks)
    raw = await _generate(
        system=system,
        prompt=prompt,
        num_predict=512,
        temperature=0.3,
        on_delta=on_delta,
    )
    answer = sanitize_answer(raw)
    if is_insufficient_answer(answer):
        return _fallback("summary")
    answer = append_sources(answer, chunks)
    return {"answer": answer, "sources": build_sources(chunks), "hit": True, "intent": "summary"}


# ===== 计算工具（AST 白名单安全求值）=====

# 二元运算白名单：只允许纯数值四则/幂/整除/取余——其余运算符一律拒绝
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# 数值绝对值上限：防 2**999999 这类「合法但灾难」的表达式把内存打爆
_MAX_ABS = 1e15
# 幂运算指数上限（同上：底数指数都合法时结果仍可能天文数字）
_MAX_POW_EXP = 64


def safe_eval_arith(expr: str) -> float:
    """AST 白名单求值：只认数字与四则/幂运算，任何名字/属性/调用/导入直接拒绝。

    为什么不让 LLM 自己算：7B 算多位数乘除会出错（幻觉会渗透到算术层）——
    计算工具存在的意义就是「路由到确定性代码」，把结果的正确性从模型手里拿回来。
    为什么 AST 而不是 eval：eval 是任意代码执行（__import__/os.system 全开），
    即使做字符串过滤也拦不住变体；AST 逐节点白名单是结构级拒绝，无绕过面。
    """
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree.body)
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ValueError("表达式结果不是数值")
    if abs(float(result)) > _MAX_ABS:
        raise ValueError("结果超出安全范围")
    return result


def _eval_node(node: ast.AST) -> float:
    """递归求值 AST 节点（仅常量/白名单运算/一元正负）。"""
    if isinstance(node, ast.Constant):
        # bool 是 int 的子类，必须显式排除（True+1=2 这类结果语义混乱）
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"非法常量: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        # 幂运算先挡大指数：2**999999 两个操作数都「合法」，但结果会撑爆内存——
        # 指数先检查再施加，非纯常量指数在递归求值时同样只会走到白名单检查
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXP:
            raise ValueError("幂指数超出安全范围")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    # 名字（含 __import__）、属性、下标、调用、推导式……统统拒绝
    raise ValueError(f"表达式包含不允许的结构: {type(node).__name__}")


# 整句算术识别：全角转半半角 + 去引导词 + 整句必须纯算术字符
_FULLWIDTH_TRANS = str.maketrans("０１２３４５６７８９＋－＊／（）．％＾", "0123456789+-*/().%^")
_ARITH_CHARS = re.compile(r"[\d\s().+\-*/%^]+")
_LEAD_WORDS = re.compile(r"^(?:请|帮我|给我|麻烦)?\s*(?:计算|算一下|求值|算)\s*[:：]?\s*")
_TRAIL_WORDS = re.compile(r"(?:等于多少|是多少|等于几|的结果|=|？|\?|。|！|!)+\s*$")


def extract_arith(question: str) -> str | None:
    """从提问中提取纯算术表达式；不是「整句即算式」就返回 None（走常规检索路由）。

    为什么判据苛刻（整句纯算术）而不是「句中含算式」：
    「第3页的公式 f(x)=3x+2 在 x=5 时等于多少」含算式片段但**意图是答疑**——
    旁路必须零误伤，宁可漏接（走 qa 也能答）不可错接。
    ^ 转成 ** ：用户按键盘 Shift+6 输入的 ^ 是数学幂记号，Python 语义是按位异或。
    """
    text = (question or "").translate(_FULLWIDTH_TRANS).strip()
    text = _LEAD_WORDS.sub("", text)
    text = _TRAIL_WORDS.sub("", text).strip()
    if not text or not _ARITH_CHARS.fullmatch(text):
        return None
    # 必须同时含数字与运算符：「2026」是年份不是算式，「+ -」没有操作数也不是
    if not re.search(r"\d", text) or not re.search(r"[+\-*/%^]", text):
        return None
    return text.replace("^", "**")


async def calc_tool(question: str, chunks: list[RetrievedChunk] | None = None) -> dict:
    """计算：AST 安全求值出确定结果 + LLM 生成步骤讲解（结果由代码定，模型只讲过程）。

    chunks 非空说明是「分类为 calc 但整句非纯算式」的路径（如「帮我算一下第3页例题」）——
    此时提取不到表达式，退回答疑工具用资料作答，路由不落空。
    """
    expr = extract_arith(question)
    if expr is None:
        if chunks:
            return await qa_tool(question, chunks)
        return _fallback("calc")
    try:
        value = safe_eval_arith(expr)
    except (ValueError, SyntaxError, OverflowError, ZeroDivisionError) as e:
        # 除零/超界等：人话兜底而不是 500——用户看到的是「算不了」不是栈
        logger.info("计算被安全求值拒绝: %s (%s)", expr, e)
        return {
            "answer": f"该表达式无法计算（{e}）。请检查是否有除零或超出范围的运算。",
            "sources": [],
            "hit": True,
            "intent": "calc",
        }
    # 展示值：整数不带 .0（1189 比 1189.0 好看），浮点保留有效位
    display = int(value) if isinstance(value, int) or float(value).is_integer() else round(value, 6)
    # 步骤讲解交给 LLM，但明确「结果已定、禁止重算」——防它把答案讲错
    steps = await gateway.generate(
        model=settings.llm_model,
        prompt=f"表达式：{expr}\n系统已计算出准确结果：{display}。"
        "请用不超过 4 步写出简要计算过程，结论必须等于系统结果，不要自行重新计算。",
        system="你是数学助教，只输出简要计算步骤，不输出多余文字。",
        num_predict=256,
        temperature=0.0,
    )
    steps = (steps or "").strip()
    answer = f"**{expr} = {display}**" + (f"\n\n{steps}" if steps else "")
    return {"answer": answer, "sources": [], "hit": True, "intent": "calc"}


# ===== 判分 =====


async def score_quiz(questions: list[dict], answers: list[str]) -> dict:
    """判分：单选代码层先判（确定性优先），简答交 LLM 按要点给分。

    返回 {"score": int, "comment": str}；LLM 输出解析失败抛 RuntimeError
    由接口层翻译成 502 人话（判分失败可重试，比给个假分数诚实）。
    """
    # 第一层：单选题精确比对——能确定的绝不交给模型不确定性
    final_scores: dict[int, int] = {}
    short_idx: list[int] = []
    for i, q in enumerate(questions):
        if q.get("type") == "choice":
            # 集合比对而非字符串相等：多选题 "BA" 与标准 "AB" 等价；
            # 单选 std={"A"} picked={"A"} 同样成立——单选是多选的特例，一套逻辑通吃
            std = {ch for ch in str(q.get("answer", "")).upper() if ch.isalpha()}
            raw_stu = answers[i] if i < len(answers) else ""
            stu = {ch for ch in str(raw_stu).upper() if ch.isalpha()}
            # 未作答（空集）恒 0 分；全对（集合相等）100
            final_scores[i] = 100 if stu and stu == std else 0
        else:
            short_idx.append(i)

    if not short_idx:
        # 全是单选：分数已确定，不产生任何 LLM 调用（零成本路径）
        score = round(sum(final_scores.values()) / len(questions)) if questions else 0
        return {"score": score, "comment": "全部为单选题，已自动判分。"}

    # 第二层：简答题（或混合卷）交 LLM——只送简答相关题，减少干扰
    sub_questions = [questions[i] for i in short_idx]
    sub_answers = [answers[i] if i < len(answers) else "" for i in short_idx]
    system, prompt = build_score_prompt(sub_questions, sub_answers)
    raw = await gateway.generate(
        model=settings.llm_model,
        prompt=prompt,
        system=system,
        num_predict=256,
        temperature=0.0,  # 判分要稳定：同一份卷子反复判分结果一致
    )
    try:
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        llm_score = int(obj["score"])
        comment = str(obj.get("comment", "")).strip()
        if not 0 <= llm_score <= 100:
            raise ValueError("分数越界")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("判分 JSON 解析失败: %s raw=%.200s", e, raw)
        raise RuntimeError("判分失败") from e
    # 总分 = 已判单选平均 + 简答 LLM 分平均（各题等权）
    total = sum(final_scores.values()) + llm_score * len(short_idx)
    score = round(total / len(questions))
    return {"score": score, "comment": comment or "已判分。"}
