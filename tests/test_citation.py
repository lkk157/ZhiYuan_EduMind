# -*- coding: utf-8 -*-
"""
溯源文本层单测（M2）：伪来源净化 + 强制来源拼接——防幻觉卖点的文本出口。

为什么这两件事必须是「代码」而不是「提示词恳求」：
7B 会忘写/编造来源；sanitize 在出口删掉一切模型自写来源，
append_sources 按真实命中强制拼接——答案末尾的【来源：文件名，第X页】
永远与真实检索结果一致，这就是「三层防线」的第 2、3 层。
"""
from app.rag.prompts import (
    FALLBACK_MESSAGE,
    append_sources,
    build_qa_prompt,
    sanitize_answer,
)
from app.rag.retriever import RetrievedChunk


def _chunk(text: str, file_name: str, page_no: int) -> RetrievedChunk:
    return RetrievedChunk(text=text, file_name=file_name, page_no=page_no, score=0.9, group_id=1)


def test_fallback_message_semantics():
    """兜底话术必须含「未找到」语义（与 /chat/ask 的 hit=false 出口口径一致）。"""
    assert "未找到" in FALLBACK_MESSAGE


def test_build_qa_prompt_contains_rules_and_materials():
    """system 四条铁律 + 资料带编号/文件名/页码 + 问题原文，一个都不能少。"""
    chunks = [_chunk("学习率过大会震荡", "讲义.docx", 3)]
    system, prompt = build_qa_prompt("学习率过大会怎样？", chunks)
    assert "不知道" in system  # 资料不足就说不知道
    assert "来源" in system  # 禁止自写来源
    assert "[1] (文件:讲义.docx, 第3页)" in prompt
    assert "学习率过大会震荡" in prompt
    assert "学习率过大会怎样？" in prompt


def test_sanitize_removes_all_fake_sources():
    """模型自写的【来源…】全删（多处/变体都删），正文原样保留。"""
    raw = "答案正文。【来源：编造.pdf，第99页】补充一句。【来源：又编的.docx，第1页】"
    assert sanitize_answer(raw) == "答案正文。补充一句。"


def test_append_sources_dedup_and_order():
    """(file_name, page_no) 去重保序拼接；同文件同页多块只记一次。"""
    chunks = [
        _chunk("a", "讲义.docx", 3),
        _chunk("b", "讲义.docx", 3),  # 同文件同页 → 去重
        _chunk("c", "讲义.docx", 5),
        _chunk("d", "习题.pdf", 1),
    ]
    out = append_sources("回答。", chunks)
    assert out == "回答。\n【来源：讲义.docx，第3页；讲义.docx，第5页；习题.pdf，第1页】"


def test_append_sources_empty_chunks_unchanged():
    """chunks 为空不追加空壳【来源：】——空壳会让用户误以为有依据。"""
    assert append_sources("回答。", []) == "回答。"


def test_fake_source_never_survives_full_pipeline():
    """端到端文本链路：伪造来源被删、真实来源被强制拼接（卖点合成验证）。"""
    chunks = [_chunk("真内容", "真讲义.docx", 7)]
    raw = "依据资料作答。【来源：幻想.pdf，第88页】"
    final = append_sources(sanitize_answer(raw), chunks)
    assert "幻想" not in final
    assert final.endswith("【来源：真讲义.docx，第7页】")
