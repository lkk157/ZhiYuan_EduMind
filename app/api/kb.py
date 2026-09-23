# -*- coding: utf-8 -*-
"""
知识库接口：分组 CRUD + 文档上传/列表/删除（M2 写路径的 HTTP 出口）。

为什么接口层仍然很薄（与 auth.py 同一约定）：
只做「参数校验 → 调 crud / ingest_file / 向量库 → 拼响应」，
增量入库状态机在 pipeline、越权过滤在 crud——接口不许重写这两样，
否则同名重传的三方一致性（DB 文档行 / 块指纹 / Chroma）迟早被漏掉一角。

三条防线在本文件的落点：
1. 路径穿越：文件名只取 Path(...).name 最后一段，上传名里的 ../ ..\\ 全部丢弃；
2. 越权探测：一切 get_* 带 user_id 过滤，非本人资源一律 NotFoundError（不泄漏存在性）；
3. 级联清理：删分组/删文档时 DB 行、块指纹、上传文件、Chroma 向量四方一起清，
   不留孤儿（孤儿向量会被检索引用出「已不存在的内容」）。
"""
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, status
from pydantic import BaseModel, Field

from app.api.auth import get_current_user
from app.core.config import PROJECT_ROOT, settings
from app.core.exceptions import AppError, NotFoundError
from app.db import crud
from app.db.models import User, split_empty_pages
from app.db.session import get_db
from app.ingest.pipeline import ingest_file
from app.rag import vector_store
from app.rag.vector_store import ChromaStore, store_for

# 导入时快照原始实现：_open_store 靠它判断 store_for 是否被单测打过补丁（与 pipeline 同款双缝，
# 原因见 pipeline._open_store 注释——兼容「patch 定义处」与「patch 调用方」两种测试姿势）
_store_for_original = store_for

# 路由前缀 /kb：知识库全部接口挂在它下面（层间契约的路由路径，禁止改动）
router = APIRouter(prefix="/kb", tags=["kb"])


def _open_store(user_id: int, group_id: int) -> ChromaStore:
    """打开「用户-分组」向量库（双缝兼容单测注入，运行时解析 store_for）。

    为什么要做双缝解析：单测必须用 EphemeralClient 版假向量库（零磁盘），
    而测试补丁的落点习惯有两种——patch 定义处（app.rag.vector_store.store_for）
    或 patch 调用方（app.api.kb.store_for）。这里优先用本模块命名空间的 store_for，
    否则回退 vector_store.store_for，两种姿势都生效，避免补丁静默失效、真 PersistentClient 落盘。
    """
    fn = store_for if store_for is not _store_for_original else vector_store.store_for
    return fn(user_id, group_id)


def _upload_dir_for(user_id: int, group_id: int) -> Path:
    """计算分组上传目录：settings.upload_dir / u{user_id} / g{group_id}（契约规定的落盘布局）。

    为什么相对路径按项目根解析：settings.upload_dir 语义是「相对项目根目录」
    （见 app/core/config.py 路径一节），若按 cwd 解析，从不同目录启动服务
    会把上传文件写得到处都是（与 vector_store._get_default_client 同一处理）。
    """
    base = settings.upload_dir
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / f"u{user_id}" / f"g{group_id}"


class GroupCreateRequest(BaseModel):
    """建分组入参。分组名唯一性由 DB 约束 + crud 双保险（撞车回 ConflictError）。"""

    name: str = Field(min_length=1, max_length=128, description="分组名")


def _require_group(db, *, user_id: int, group_id: int):
    """取本人分组，取不到（不存在或非本人）一律 NotFoundError。

    为什么统一报「不存在」而不是 403：报 403 等于承认「这个 gid 存在，只是不归你」，
    攻击者可借此枚举全站分组 id——防探测是越权防线的一部分。
    """
    group = crud.get_kb_group(db, user_id=user_id, group_id=group_id)
    if group is None:
        raise NotFoundError("分组不存在")
    return group


@router.post("/groups", status_code=status.HTTP_201_CREATED)
def create_group(
    body: GroupCreateRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """新建知识库分组。同名分组撞车由 crud 抛 ConflictError「分组名已存在」。"""
    group = crud.create_kb_group(db, user_id=user.id, name=body.name)
    return {"id": group.id, "name": group.name}


@router.get("/groups")
def list_groups(
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """列出本人全部分组（含文档数），供前端知识库首页与提问时的分组多选框共用。"""
    groups = crud.list_kb_groups(db, user_id=user.id)
    # doc_count 逐组统计：分组数与文档数都是小规模（个人知识库），N+1 在这里可接受；
    # 聚合 SQL 留给 crud 层演进，接口不写 SQL（分层约定）
    return [
        {
            "id": g.id,
            "name": g.name,
            "created_at": g.created_at,
            "doc_count": len(crud.list_documents(db, user_id=user.id, group_id=g.id)),
        }
        for g in groups
    ]


@router.delete("/groups/{gid}")
def delete_group(
    gid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """删除分组，级联清空：DB 行（crud 连带文档+指纹）+ 上传目录树 + 对应 Chroma collection。

    为什么先清衍生资源、最后删 DB 行：DB 是事实源放最后一步——
    中途失败时衍生清理是幂等的（rmtree 容错 / drop 后可重建），
    重试删除即可收敛；反过来先删 DB 再清失败，用户会看到报错却再也找不到分组入口。
    """
    _require_group(db, user_id=user.id, group_id=gid)
    # 1) 向量库整库销毁：分组是物理隔离边界，删组即删 collection，知识不串味也不留孤儿索引
    _open_store(user.id, gid).drop()
    # 2) 上传目录树连根删（ignore_errors：目录不存在/已删过都视作清理完成，删除必须可重试）
    shutil.rmtree(_upload_dir_for(user.id, gid), ignore_errors=True)
    # 3) DB 级联删（crud 内部按 指纹 → 文档 → 分组 顺序手工级联，兼容 SQLite/MySQL）
    crud.delete_kb_group(db, user_id=user.id, group_id=gid)
    return {"ok": True}


@router.get("/groups/{gid}/documents")
def list_documents(
    gid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """列出分组下全部文档（empty_pages 还原成 list[int] 契约字段）。"""
    _require_group(db, user_id=user.id, group_id=gid)
    docs = crud.list_documents(db, user_id=user.id, group_id=gid)
    return [
        {
            "id": d.id,
            "file_name": d.file_name,
            "status": d.status,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            # 库里存逗号拼接字符串，出口还原成页码列表（编码/还原见 models.join/split_empty_pages）
            "empty_pages": split_empty_pages(d.empty_pages),
            "updated_at": d.updated_at,
        }
        for d in docs
    ]


@router.post("/groups/{gid}/documents")
async def upload_document(
    gid: int,
    file: UploadFile,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """上传并同步入库一个文件（同名重传 = chunk 级增量更新，由 ingest_file 状态机处理）。

    为什么同步 await 入库而不是先返回再后台跑：入库结果（added/changed/skipped_identical）
    是用户上传后的即时反馈，前端要据此提示「已跳过/新增 N 块」；后台化属后续优化，
    现阶段同步链路简单可测，也避免引入任务队列依赖（CLAUDE.md 禁 Docker/Redis）。
    """
    _require_group(db, user_id=user.id, group_id=gid)

    # ★ 路径穿越防线：只取文件名最后一段，上传名里的 ../../secret.txt 之类全部丢弃
    file_name = Path(file.filename or "").name
    if not file_name:
        # 空文件名多半是异常客户端/脚本直打接口，统一业务错误出口（400）
        raise AppError("文件名不能为空", code=400)

    # 落盘契约布局：upload_dir/u{user_id}/g{group_id}/文件名
    target_dir = _upload_dir_for(user.id, gid)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file_name
    # 二进制原样落盘（pdf/docx/pptx 都是二进制容器，绝不能走文本读写）
    target_path.write_bytes(await file.read())

    # 同步增量入库：skipped_identical / added / changed / removed 一律原样透传（契约字段）
    result = await ingest_file(
        db,
        user_id=user.id,
        group_id=gid,
        file_name=file_name,
        file_path=target_path,
    )
    return {
        "doc_id": result.doc_id,
        "skipped_identical": result.skipped_identical,
        "added": result.added,
        "changed": result.changed,
        "removed": result.removed,
        "unchanged": result.unchanged,
        "page_count": result.page_count,
        "chunk_count": result.chunk_count,
        "empty_pages": result.empty_pages,
    }


@router.delete("/documents/{did}")
def delete_document(
    did: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """删除文档，级联清空：DB 行+块指纹 + 该文档全部向量 + 物理文件。

    为什么必须先读指纹再删库：向量 id 是 "{document_id}:{chunk_index}"，
    chunk_index 清单只存在于指纹表里；先把 id 清单取到手，才能把 Chroma 里的块删干净，
    否则孤儿向量会继续被检索命中（引用出「已不存在的资料」，污染溯源）。
    """
    doc = crud.get_document(db, user_id=user.id, doc_id=did)
    if doc is None:
        # 非本人/不存在统一 404，防探测（理由同 _require_group）
        raise NotFoundError("文档不存在")

    # 1) 取当前全部块指纹 → 拼成向量 id 清单（此时指纹还在，删库后就拼不出来了）
    fingerprints = crud.list_chunk_fingerprints(db, document_id=doc.id)
    vector_ids = [f"{doc.id}:{fp.chunk_index}" for fp in fingerprints]
    # 2) 删向量（幂等：清单为空/no-op 都安全）
    _open_store(user.id, doc.group_id).delete(vector_ids)
    # 3) 删 DB 行（crud 连带删块指纹行，不留孤儿指纹导致下次同名重建误判增量）
    crud.delete_document(db, user_id=user.id, doc_id=did)
    # 4) 删物理文件（先判存在：同名重传共用一个落盘路径，已被清理时跳过即可）
    file_path = Path(doc.file_path)
    if file_path.is_file():
        file_path.unlink()
    return {"ok": True}
