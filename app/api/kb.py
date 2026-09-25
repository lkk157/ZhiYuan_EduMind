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
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, Response, status
from pydantic import BaseModel, Field

from app.api.auth import get_current_user
from app.core.config import PROJECT_ROOT, settings
from app.core.exceptions import AppError, NotFoundError
from app.db import crud
from app.db.models import User, split_empty_pages
from app.db.session import get_db
from app.ingest.parsers import SUPPORTED_SUFFIXES
from app.ingest.pipeline import ingest_file
from app.rag import vector_store
from app.rag.vector_store import ChromaStore, store_for

# 导入时快照原始实现：_open_store 靠它判断 store_for 是否被单测打过补丁（与 pipeline 同款双缝，
# 原因见 pipeline._open_store 注释——兼容「patch 定义处」与「patch 调用方」两种测试姿势）
_store_for_original = store_for

# 路由前缀 /kb：知识库全部接口挂在它下面（层间契约的路由路径，禁止改动）
router = APIRouter(prefix="/kb", tags=["kb"])

logger = logging.getLogger(__name__)


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
async def upload_documents(
    gid: int,
    file: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """批量上传并逐个同步入库（产品需求：单批 ≤MAX_UPLOAD_FILES、总大小 ≤MAX_UPLOAD_TOTAL_MB）。

    响应契约（2026-09-24 批量化改版；单文件调用返回同结构、results 只有一项）：
    {"results":[{file_name, ok, error?, ...IngestResult 字段}], "succeeded": n, "failed": m}

    为什么批次校验先于任何处理：超限整批拒绝并明确提示，好过处理一半再报错
    （用户面对「传 5 个好了 3 个」的半截结果无从下手）；
    为什么逐文件顺序处理、单文件失败不连坐：OCR/推理本就经 gateway 串行，
    顺序处理显存与内存都友好（一次只读当前文件，200MB 上限也不会整批驻留内存）；
    一个坏文件不该让同批其余文件白传。同名重传 = chunk 级增量（ingest_file 状态机，逐文件独立）。
    """
    _require_group(db, user_id=user.id, group_id=gid)

    # --- 批次级校验（先于任何落盘/入库；客户端限制不是安全边界，服务端必须复校）---
    if not file:
        raise AppError("未收到任何文件", code=400)
    if len(file) > settings.max_upload_files:
        raise AppError(f"单批最多上传 {settings.max_upload_files} 个文件", code=400)
    # UploadFile.size 由 Starlette 解析 multipart 时填充，无需整读文件即可校验总大小
    total_bytes = sum(int(getattr(f, "size", 0) or 0) for f in file)
    limit_bytes = settings.max_upload_total_mb * 1024 * 1024
    if total_bytes > limit_bytes:
        raise AppError(f"本批文件总大小超过 {settings.max_upload_total_mb}MB 上限", code=400)

    target_dir = _upload_dir_for(user.id, gid)
    target_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for f in file:
        # ★ 路径穿越防线：只取文件名最后一段（逐文件独立处理）
        file_name = Path(f.filename or "").name
        try:
            if not file_name:
                raise AppError("文件名不能为空", code=400)
            # 格式白名单前置到落盘之前：不支持的类型不落盘、不留文档行
            # （这类文件重传永远不会成功，走 ingest 的失败留痕只会污染文档列表）
            if Path(file_name).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise AppError("不支持的文件类型（仅支持 PDF/Word/PPT/图片）", code=400)
            target_path = target_dir / file_name
            # 二进制原样落盘（每次只读当前文件，处理完即释放）
            target_path.write_bytes(await f.read())
            result = await ingest_file(
                db, user_id=user.id, group_id=gid, file_name=file_name, file_path=target_path
            )
            results.append(
                {
                    "file_name": file_name,
                    "ok": True,
                    "error": None,
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
            )
        except AppError as e:
            # 业务层可预期错误（不支持的类型/名称非法/Ollama 连不上等）：人话直出，继续下一份
            results.append({"file_name": file_name, "ok": False, "error": e.message})
        except Exception:
            # 非预期异常（解析器崩溃等）：pipeline 已把文档标 failed 留痕，
            # 这里翻译成人话并继续处理同批其余文件，完整栈进日志排查
            logger.exception("批量上传中单文件入库失败: %s", file_name)
            results.append({"file_name": file_name, "ok": False, "error": "入库失败，请重试"})

    succeeded = sum(1 for r in results if r["ok"])
    return {"results": results, "succeeded": succeeded, "failed": len(results) - succeeded}


# 支持原页预览的图片后缀 → media type（图片文件直出字节，不重渲染）
_IMAGE_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@router.get("/groups/{gid}/page-image")
def page_image(
    gid: int,
    file_name: str,
    page_no: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """渲染文档某一页为 PNG（体验增强包：来源卡片「查看原页」的数据源）。

    为什么 PDF 用 pymupdf 2x 渲染而不是复用入库时的 OCR 图：入库图只覆盖
    空文本页，而「查看原页」任何页都可能被点（用户就是要看原件）；2x 缩放是
    渲染清晰度与响应体积的平衡（答辩演示走本机回环，几十 ms 足够）。

    为什么 Word/PPT 直接人话 404：Word 无稳定分页（入库是逻辑页概念）、
    PPT 幻灯片渲染另有一套——强行对齐页码会显示错误的页面（比不显示更糟），
    「如实说不支持」好过「给个错的」。
    越权口径与其余 kb 端点一致：分组/文档任一非本人 → 404 防探测。
    """
    _require_group(db, user_id=user.id, group_id=gid)
    doc = crud.get_document_by_name(db, user_id=user.id, group_id=gid, file_name=file_name)
    if doc is None:
        raise NotFoundError("文档不存在")
    path = Path(doc.file_path)
    if not path.is_file():
        raise NotFoundError("文件已不存在")

    suffix = path.suffix.lower()
    if suffix in _IMAGE_MEDIA:
        # 图片原件直出（入库的就是它本身，页码=1）
        if page_no != 1:
            raise NotFoundError("页码超出文档范围")
        return Response(content=path.read_bytes(), media_type=_IMAGE_MEDIA[suffix])

    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(str(path)) as pdf:
            if not 1 <= page_no <= pdf.page_count:
                raise NotFoundError("页码超出文档范围")
            # 2x 矩阵渲染：72dpi 原始渲染在高分屏上发虚（OCR 阶段同款取舍）
            pix = pdf[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(2, 2))
            return Response(content=pix.tobytes("png"), media_type="image/png")

    raise AppError("该格式暂不支持原页预览（支持 PDF 与图片）", code=404)


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
