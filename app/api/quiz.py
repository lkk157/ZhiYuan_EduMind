# -*- coding: utf-8 -*-
"""
错题接口（M5）：错题本列表（带知识点图谱关联推荐）+ 删除。

判分落库不在这里——它长在 /chat/score 上（判分瞬间手里才有题/答案/对错/教材锚，
就地写入零额外请求）；本文件只服务「错题与学情」页的读与删。

关联推荐的计算时机：列表请求时**现算**（文件名解析+存量向量查询，毫秒级）——
动态图谱的产品语义由此保证：新上传的课件在下次打开列表时自动入图，
没有任何需要同步的缓存表。
"""
from fastapi import APIRouter, Depends

from app.api.auth import get_current_user
from app.core.exceptions import NotFoundError
from app.db import crud
from app.db.models import User
from app.db.session import get_db
from app.memory import knowledge_graph

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.get("/records")
def list_records(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """本人错题列表（仅答错的），每题附知识点图谱关联推荐。

    推荐按 ref_file 分组算一次复用（同章节的多道错题共享图查询——
    逐题重算是 O(错题数×图) 的浪费，本体量下也该省）。
    """
    records = crud.list_quiz_records(db, user_id=user.id, limit=50)

    # 图谱输入（每次现取 = 动态：新上传文档立刻在列）
    file_names = []
    group_ids = []
    for group in crud.list_kb_groups(db, user_id=user.id):
        group_ids.append(group.id)
        file_names.extend(d.file_name for d in crud.list_documents(db, user_id=user.id, group_id=group.id))

    related_cache: dict[str, list[dict]] = {}
    items = []
    for r in records:
        if r.ref_file and r.ref_file not in related_cache:
            # 同一 ref_file 的推荐算一次（结构边零成本；语义边用存量向量，毫秒级）
            related_cache[r.ref_file] = [
                {
                    "file_name": it.file_name,
                    "label": it.label,
                    "relation": it.relation,
                    "score": it.score,
                }
                for it in knowledge_graph.related_for(user.id, r.ref_file, file_names, group_ids)
            ]
        items.append(
            {
                "id": r.id,
                "question": r.question,
                "question_type": r.question_type,
                "user_answer": r.user_answer,
                "correct_answer": r.correct_answer,
                "explanation": r.explanation,
                "ref_file": r.ref_file,
                "ref_page": r.ref_page,
                "created_at": r.created_at,
                "related": related_cache.get(r.ref_file, []),
            }
        )
    return items


@router.delete("/records/{rid}")
def delete_record(
    rid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """删除一条错题（非本人/不存在 → 404 防探测，口径与其余资源一致）。"""
    if not crud.delete_quiz_record(db, user_id=user.id, record_id=rid):
        raise NotFoundError("错题记录不存在")
    return {"ok": True}
