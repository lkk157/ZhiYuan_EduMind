# -*- coding: utf-8 -*-
"""
问答提示词编排与答案净化（防幻觉的关键文本出口）。

为什么提示词要独立成层：
1. 「只依据资料回答 + 不许自己写来源」是溯源 RAG 的命门——
   模型自写的来源是编造的，必须在文本上禁掉、在输出上删掉，双保险；
2. 真正的来源标注由 append_sources 按检索结果强制统一追加，
   保证答案末尾的【来源：文件名，第X页】永远与真实命中一致；
3. 提示词调整（答辩前微调措辞）只动本文件，不动检索/接口层。
"""
import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # 只为类型标注：prompts 是纯文本层，运行期不依赖检索层
    from app.rag.retriever import RetrievedChunk


# 兜底话术：召回为空时由调用方直接返回给用户（此时严禁调用 LLM，防幻觉）；
# 「检索到但模型判定资料不足」的第二条兜底路径也复用同一话术（口径统一，见 is_insufficient_answer）
FALLBACK_MESSAGE = "知识库中未找到相关内容。请换一种问法，或先上传相关课件到知识库。"


# ===== 模型「资料不足」判定（2026-09-24 产品需求：不知道就不输出来源）=====
#
# 为什么需要这道闸门：system 第 2 条要求模型「资料不足就说不知道」——模型守了规矩，
# 但 append_sources 只看「检索有没有块」，照样强拼【来源】，出现
# 「嘴上说不知道、脚下引一串出处」的自相矛盾。判定必须发生在「生成后、拼来源前」。

# 主判据：system 要求的固定句式开头（提示词配合代码，确定性最高）
_INSUFFICIENT_PREFIX = "根据现有资料无法回答"

# 副判据：模型没按固定句式时的兜底短语——刻意收窄到「指向资料缺失」的强特征，
# 绝不放裸的「不知道」（会误杀「很多同学不知道这个原理，正确的是…」这类带实质内容的正常答案）
_INSUFFICIENT_PHRASES = (
    "根据现有资料无法回答",
    "资料中未提及",
    "资料未提及",
    "未在资料中",
    "资料中没有提到",
    "资料没有提到",
    "无法根据现有资料",
)


def is_insufficient_answer(answer: str) -> bool:
    """判断模型的回答是否在表达「资料不足/答不上」——True 则调用方走兜底口径、不拼来源。

    判定层次（防误杀是本函数的第一设计目标）：
    1. 空答案 True：模型吐了空串，拼个光秃秃的【来源】更荒唐，按没答上处理；
    2. 主判据：以固定句式开头（system 明确要求的输出格式）直接 True；
    3. 副判据：只在**第一句话**里找强特征短语——
       「学习率过大会震荡。但资料中未提及具体数值。」这种先给答案再补注的正常回答，
       短语出现在第二句，不会被误判；而「很抱歉，资料中未提及…」短语在首句，会被兜住。

    为什么不做语义级判断（再调一次 LLM 判「这是不是不知道」）：为判「不知道」而多打一次
    7B 推理，性价比为负还碰显存串行队列——正则强特征 + 提示词配合对固定话术已足够稳，
    漏网的擦边案例由 SCORE_THRESHOLD 标定（M2 遗留待办）从检索侧根治。
    """
    text = (answer or "").strip()
    if not text:
        return True
    if text.startswith(_INSUFFICIENT_PREFIX):
        return True
    # 首句 = 第一个句号/叹号/问号/换行之前的部分（没有标点则整段视为首句）
    first_sentence = re.split(r"[。！？!?\n]", text, maxsplit=1)[0]
    return any(phrase in first_sentence for phrase in _INSUFFICIENT_PHRASES)


# 【来源…】片段（全角方括号包裹、以「来源」开头）= 模型自写的伪溯源。
# 为什么必须删：来源只能由 append_sources 按真实检索结果统一追加，
# 模型自己写的来源是幻觉，留着会污染溯源（防幻觉红线）。
_SOURCE_PATTERN = re.compile(r"【来源[^】]*】")


def build_qa_prompt(
    question: str, chunks: Sequence["RetrievedChunk"], *, guide: bool = False
) -> tuple[str, str]:
    """把问题与召回资料编排成 (system, prompt) 两段提示词。

    资料按 `[n] (文件:xxx, 第X页)` 带编号编排：
    - 编号让模型可以「据资料[2]…」地引用，回答有依据感；
    - 文件名+页码随资料一起给模型，它才可能说出「第X页讲了…」这类可核对的话，
      但来源标注本身不许它写（由系统统一追加）。

    guide=True（M4 引导式答疑/苏格拉底模式）：先反问引导再给提示，不直接端出完整答案——
    教育学「脚手架」理念的落点，产品开关默认关闭（不影响既有答疑行为）。
    引导模式仍受「资料不足固定句式」铁律约束（is_insufficient_answer 判定不受影响）。

    注意：chunks 为空时调用方不应调 LLM（应直接回 FALLBACK_MESSAGE），
    本函数不负责兜底话术——文本层不掺业务分支。
    """
    # system 明确「只依据资料 / 不足输出固定句式 / 禁止编造 / 禁止自写来源」四条铁律。
    # 第 2 条为什么要求固定句式开头：代码层 is_insufficient_answer 按该前缀判定
    # 「模型不知道」→ 走兜底、不拼来源——提示词给确定性锚点，判定才不是猜谜。
    system = (
        "你是「知源」教育知识库问答助手。回答必须遵守：\n"
        "1. 只依据用户消息中给出的参考资料回答，禁止使用资料之外的知识；\n"
        "2. 资料不足以回答该问题时，回答必须以「根据现有资料无法回答」开头，"
        "其后可一句话说明缺少哪方面的资料，禁止编造、禁止猜测；\n"
        "3. 禁止自己书写任何来源标注（例如【来源：…】）——系统会在答案末尾统一追加来源，"
        "自己写的来源一律会被删除；\n"
        "4. 使用与提问相同的语言回答（默认中文），面向学生，条理清晰。"
    )
    if guide:
        # 追加而非替换：铁律 1-4 原样保留，引导式只是「怎么答」的风格开关
        system += (
            "\n5.【引导式答疑模式】不要直接给出完整最终答案：先用一个切中要害的"
            "引导性反问或分步提示启发学生自己思考，最后用一句话点到关键结论即可；"
            "若学生的问题本身已包含完整推理，指出其正确/错误的那一步即可。"
        )

    # 资料块：[n] (文件:xxx, 第X页) + 正文，编号从 1 起
    blocks: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{n}] (文件:{chunk.file_name}, 第{chunk.page_no}页)\n{chunk.text}")
    materials = "\n\n".join(blocks)

    prompt = (
        "请只依据以下参考资料回答问题。\n\n"
        f"{materials}\n\n"
        f"问题：{question}\n"
        "回答："
    )
    return system, prompt


def build_summary_prompt(question: str, chunks: Sequence["RetrievedChunk"]) -> tuple[str, str]:
    """总结任务提示词：只依据资料做结构化小结（M4 总结工具用）。

    为什么单独一套而不是复用问答模板：总结的输出形态是「要点列表」，
    与问答的「针对问题作答」不同——给模型明确的结构指令，输出才稳定可展示；
    铁律（不足固定句式/禁自写来源）与问答完全一致，判定与净化逻辑可整套复用。
    """
    system = (
        "你是「知源」教育知识库的学习总结助手。回答必须遵守：\n"
        "1. 只依据用户消息中给出的参考资料做总结，禁止补充资料之外的内容；\n"
        "2. 资料不足以完成总结时，回答必须以「根据现有资料无法回答」开头，禁止编造；\n"
        "3. 禁止自己书写任何来源标注——系统会统一追加；\n"
        "4. 用与提问相同的语言（默认中文），以「要点列表」形式输出，"
        "每条要点一行、不超过 40 字，面向学生复习场景；\n"
        "5.【范围纪律】只总结资料实际覆盖的范围：第一行先写「覆盖范围：文件名（第X-Y页）」，"
        "资料跨多个文件或页码段时按来源分段列要点，"
        "禁止把范围外或不相关的内容混进总结、禁止泛泛而谈整门课。"
    )
    blocks: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{n}] (文件:{chunk.file_name}, 第{chunk.page_no}页)\n{chunk.text}")
    materials = "\n\n".join(blocks)
    prompt = (
        "请只依据以下参考资料完成总结。\n\n"
        f"{materials}\n\n"
        f"总结要求：{question}\n"
        "输出："
    )
    return system, prompt


# 出题任务的试题 JSON 契约（前后端共同遵守，改字段必须三端同步）：
# {"type":"quiz","questions":[
#   {"type":"choice","question":"...","options":["A. ..","B. ..","C. ..","D. .."],
#    "answer":"A","explanation":"..."},
#   {"type":"short","question":"...","answer":"标准答案","explanation":"..."}]}
_QUIZ_SCHEMA = (
    '{"type":"quiz","questions":['
    '{"type":"choice","question":"题干","options":["选项A","选项B","选项C","选项D"],'
    '"answer":"A","explanation":"解析"},'
    '{"type":"short","question":"题干","answer":"标准答案","explanation":"解析"}]}'
)


def build_quiz_prompt(question: str, chunks: Sequence["RetrievedChunk"]) -> tuple[str, str]:
    """出题任务提示词：依据资料生成结构化试题 JSON（M4 出题/随堂测工具用）。

    为什么强制 JSON 输出而不是自然语言：试题要渲染成交互卡片、要判分，
    结构化才能被代码可靠解析（parse 失败有兜底，但要尽量一次成）；
    道数不写死在模板里——用户说「出3道」「来5道随堂测」由模型从 question 里读，
    一个模板覆盖两种场景（含随堂测），不用为道数再造参数。
    """
    system = (
        "你是「知源」教育知识库的出题人。生成规则（题型与道数必须服从用户要求）：\n"
        "1. 只依据给出的参考资料出题，题目必须能在资料中找到答案，禁止编造知识点；\n"
        "2.【题型服从】用户要求算法题/编程题/计算题/简答题/问答题时，必须出 short 类型"
        "（题干要求写出思路、代码或计算过程），禁止降级成选择题；"
        "用户明确要求多选题时才出多选（answer 为多个字母，如 \"AB\"），"
        "否则所有 choice 一律单选、恰好一个正确选项；\n"
        "3.【道数服从】用户说几道就出几道，一个不少一个不多；未指定时出 3 道；\n"
        "4. 选择题四个选项，三个干扰项必须来自该知识点的常见误解；"
        "正确选项位置随机分布，禁止总是 A；\n"
        "5. 只输出一个 JSON 对象，不要输出任何解释、前后缀或 Markdown 代码块标记；\n"
        f"6. JSON 结构必须是：{_QUIZ_SCHEMA}\n"
        "其中 choice 的 answer 是正确选项字母（单选为一个字母），short 的 answer 是标准答案文本。\n"
        '算法题示例：{"type":"short","question":"用伪代码写出求最大公约数的辗转相除法，'
        '并说明其时间复杂度。","answer":"gcd(a,b)=gcd(b,a%b)，直到 b 为 0 返回 a；'
        '时间复杂度 O(log min(a,b))","explanation":"辗转相除法基于余数递降，每步规模至少减半。"}'
    )
    blocks: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{n}] (文件:{chunk.file_name}, 第{chunk.page_no}页)\n{chunk.text}")
    materials = "\n\n".join(blocks)
    prompt = (
        "请只依据以下参考资料出题。\n\n"
        f"{materials}\n\n"
        f"出题要求：{question}\n"
        "JSON："
    )
    return system, prompt


def build_score_prompt(questions: list[dict], answers: list[str]) -> tuple[str, str]:
    """判分提示词：学生作答 vs 标准答案 → {"score":0-100,"comment":"..."}。

    为什么判分也让 LLM 做而不是字符串比对：简答题的「意思对了但措辞不同」
    是常态（教育场景的正确答案本就开放），机械比对会把对的判成错的；
    单选题 answer 精确匹配在代码层先做（见 score 端点），LLM 只判简答——
    能确定的绝不确定性，是判分设计的第一原则。
    """
    system = (
        "你是阅卷老师。对照标准答案给学生的作答打分，只输出一个 JSON 对象："
        '{"score":0到100的整数,"comment":"不超过50字的中文评语"}，'
        "不要输出任何其他内容。单选题答对得满分、答错 0 分；简答题按要点给分。"
    )
    lines = []
    # 标准答案从题目 dict 里取（q["answer"]），序列只配对「题面 ↔ 学生作答」两个维度
    for i, (q, stu) in enumerate(zip(questions, answers), start=1):
        lines.append(
            f"第{i}题：{q.get('question', '')}\n"
            f"  标准答案：{q.get('answer', '')}\n"
            f"  学生作答：{stu or '（未作答）'}"
        )
    prompt = "\n".join(lines)
    return system, prompt


def sanitize_answer(answer: str) -> str:
    """删掉模型自写的一切【来源…】片段，返回净化后的答案。

    为什么必须净化：提示词只是「劝」，模型仍可能偷写来源（幻觉重灾区），
    出口处用正则强制删除，保证最终来源只有 append_sources 追加的那份是真的。
    """
    return _SOURCE_PATTERN.sub("", answer).strip()


def append_sources(answer: str, chunks: Sequence["RetrievedChunk"]) -> str:
    """在答案末尾强制追加【来源：文件名A，第X页；文件名B，第Y页】。

    规则（契约）：
    - (file_name, page_no) 去重保序：同文件同页的多个块只记一次，
      顺序跟检索得分顺序一致（最重要的来源排最前）；
    - chunks 为空则不追加——空壳【来源：】会让用户误以为有依据，宁可没有。
    """
    parts: list[str] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.file_name, chunk.page_no)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{chunk.file_name}，第{chunk.page_no}页")
    if not parts:
        return answer
    return f"{answer.rstrip()}\n【来源：{'；'.join(parts)}】"
