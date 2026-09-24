# -*- coding: utf-8 -*-
"""
OCR 回填层（M3 多模态入库核心）：两类任务都经 GLM-OCR 转文本——
A) 空页（扫描页/纯图页/图片文件）：整页/整图识别 → **写回**该页文本；
B) 混合页嵌入图（同一页既有文字又有图）：抽页内嵌入图片识别 → **追加**到该页文字尾部
   （文字层照常直读，绝不整页重复渲染 OCR——防止同一页出现两份不一致的文本）。

★ 显存互斥序列（CLAUDE.md §3 红线在本阶段的代码落点，答辩重点讲这里）：
1. 先查 /api/ps 确认 LLM 是否驻留——驻留才卸载（对未加载模型发 keep_alive=0
   会让 Ollama 先加载再卸载，白等几十秒）；
2. ★ 批次驻留策略：一张图一次「加载→卸载」会让多图文档慢得离谱（每次冷加载数秒），
   所以批内用短驻留（_BATCH_HOLD=5m）共享同一次加载，批末显式 unload 兑现
   settings.ocr_keep_alive=0 的「用完即卸」——红线语义从「每次调用后卸」精确为
   「本批（=本次上传）用完即卸」，时长有界（最坏 5 分钟自动过期）；
3. 全程经 gateway（排队闸门 + 互斥锁），与任何生成调用天然串行；
4. OCR 全部结束后流水线才开始向量化——时序上「OCR 与 LLM 时段隔离」自动成立。

为什么 OCR 失败不炸整个入库：文本层内容仍有价值（一页识别失败不该连坐全部已解析内容），
空页失败码保留在 empty_pages 返回给调用方——用户重新上传即可重试（文件级短路对此例外）。

页图像来源按格式分发（与 parsers 的「格式差异收口」同一思路）：
- 图片文件：文件字节本身就是图（走 A 类）；
- PDF 空页：pymupdf 整页渲染成 PNG；PDF 有字页：抽嵌入位图（≥_EMBED_MIN_EDGE 过滤 logo/装饰）；
- PPT：空页取该幻灯片全部图片；有字页同样抽内嵌图片（多图逐张识别，全档去重防复用水印重复烧显存）；
- Word：无页的图像概念，图片识别列为已知限制（见 ROADMAP 风险表）。
"""
import base64
import hashlib
import logging
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import UpstreamError
from app.core.llm import gateway
from app.ingest.parsers import ParsedPage

logger = logging.getLogger(__name__)

# OCR 任务提示：GLM-OCR 是「图 → Markdown」专职模型，指令要求逐字转换不加解释
OCR_PROMPT = "请将图像中的全部文字、公式与表格按原文顺序转换为 Markdown 文本，不要添加任何解释。"

# 整图 OCR 的输出上限：比生成类默认 512 大（一页文档轻松过 500 token），
# 但仍受控（红线要求输出长度有界）；temperature=0 保证识别确定性
OCR_NUM_PREDICT = 2048

# 混合页嵌入图识别文本的拼接标记：进切块后用户能看出「这段来自该页的图」
EMBEDDED_IMAGE_MARKER = "[图片内容]"

# 嵌入图最小边长（像素）：小于它的视作 logo/装饰/分隔线，识别了也只是噪声，不值得烧显存
_EMBED_MIN_EDGE = 200

# 批内驻留时长：多图批次共享同一次模型加载（理由见模块 docstring 的红线第 2 条）
_BATCH_HOLD = "5m"


def _to_b64(data: bytes) -> str:
    """字节流转 base64 ASCII 串（Ollama images 参数格式，不带 data: 前缀）。"""
    return base64.b64encode(data).decode("ascii")


def _pptx_picture_b64s(file_path: Path, page_no: int) -> list[str]:
    """取 PPT 某幻灯片全部内嵌图片的 base64（空列表 = 该页没有图片）。"""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(str(file_path))
    if page_no - 1 >= len(prs.slides):
        return []
    out: list[str] = []
    for shape in prs.slides[page_no - 1].shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            out.append(_to_b64(shape.image.blob))
    return out


def _pdf_embedded_b64s(file_path: Path, page_no: int) -> list[str]:
    """抽 PDF 某页的嵌入位图（≥_EMBED_MIN_EDGE 尺寸过滤），**不做整页渲染**。

    为什么有字页绝不能整页渲染：文字层已经直读入库，再渲染 OCR 一遍会得到
    两份可能不一致的同页文本（重复且污染溯源）；只抽图才与文字层互补。
    """
    import pymupdf

    doc = pymupdf.open(str(file_path))
    try:
        if page_no - 1 >= doc.page_count:
            return []
        page = doc.load_page(page_no - 1)
        out: list[str] = []
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                extracted = doc.extract_image(xref)
            except Exception:
                continue  # 单张抽取失败跳过，不连坐该页其他图片
            if extracted.get("width", 0) < _EMBED_MIN_EDGE or extracted.get("height", 0) < _EMBED_MIN_EDGE:
                continue  # 小图 = 装饰/logo，识别出来是噪声
            out.append(_to_b64(extracted["image"]))
        return out
    finally:
        doc.close()


def _page_images_b64(file_path: Path, page_no: int) -> list[str]:
    """A 类（空页）取「整页/整图」来源：图片文件自身 / PDF 整页渲染 / PPT 全部图片。"""
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".png", ".jpg", ".jpeg"}:
            return [_to_b64(file_path.read_bytes())]

        if suffix == ".pdf":
            # pymupdf 渲染该页为 PNG（2 倍缩放：72dpi 的渲染对 OCR 太糊，2x 是质量/耗时平衡点）
            import pymupdf

            doc = pymupdf.open(str(file_path))
            try:
                if page_no - 1 >= doc.page_count:
                    return []
                pix = doc.load_page(page_no - 1).get_pixmap(matrix=pymupdf.Matrix(2, 2))
                return [_to_b64(pix.tobytes("png"))]
            finally:
                doc.close()

        if suffix == ".pptx":
            return _pptx_picture_b64s(file_path, page_no)
    except Exception:
        # 取图失败（文件损坏等）视作「无图像来源」→ 该页保留在 empty_pages
        logger.exception("取页图像失败: %s 第%s页", file_path.name, page_no)
        return []

    return []  # Word 等无页图像格式


def _embedded_images_per_page(file_path: Path, pages: list[ParsedPage], empty_set: set[int]) -> dict[int, list[str]]:
    """B 类（混合页）收集：{有字页页码: [嵌入图 b64...]}，全档去重（同图只识别一次）。

    全档去重的原因：课件常见「同一 logo/水印图复用几十页」，不去重就是几十次
    白烧显存的重复识别；按图像字节哈希去重，复用图只认第一处。
    """
    suffix = file_path.suffix.lower()
    if suffix not in {".pdf", ".pptx"}:
        return {}  # 图片文件/Word 没有「页内嵌入图」概念

    result: dict[int, list[str]] = {}
    seen: set[str] = set()
    try:
        for page in pages:
            if page.page_no in empty_set:
                continue  # 空页已走 A 类整页识别，避免同一张图认两遍
            if suffix == ".pdf":
                images = _pdf_embedded_b64s(file_path, page.page_no)
            else:
                images = _pptx_picture_b64s(file_path, page.page_no)
            fresh = []
            for b64 in images:
                key = hashlib.sha256(b64.encode("ascii")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                fresh.append(b64)
            if fresh:
                result[page.page_no] = fresh
    except Exception:
        # 嵌入图收集失败不阻断主流程（文字层照常入库），日志留痕排查
        logger.exception("收集页内嵌入图失败: %s", file_path.name)
    return result


async def ocr_fill_pages(
    file_path: Path, pages: list[ParsedPage], empty_pages: list[int]
) -> list[int]:
    """执行 A+B 两类 OCR 并把文本写回 ParsedPage，返回「仍失败」的空页页码。

    - A 类空页识别成功 → 页文本**写回**，移出 empty_pages（进向量库）；
    - A 类失败/无图 → 保留在返回清单（前端提示、可重传重试）；
    - B 类混合页嵌入图识别成功 → 文本**追加**到该页文字尾部（EMBEDDED_IMAGE_MARKER 标注），
      失败只记日志（该页文字层已入库，不产生失败账）。
    零任务短路：既无空页图也无嵌入图时（纯文本文档）完全不碰显存——
    不查驻留、不卸模型、不发识别请求。
    """
    pages_by_no = {p.page_no: p for p in pages}
    empty_set = set(empty_pages)

    # ---------- 收集任务（纯本地，零模型调用）----------
    render_tasks: list[tuple[int, list[str]]] = []  # A 类：空页整页/整图
    still_failed: list[int] = []
    for pn in empty_pages:
        images = _page_images_b64(file_path, pn)
        if images:
            render_tasks.append((pn, images))
        else:
            still_failed.append(pn)  # 无图像来源（纯空白页/Word 逻辑页）
    embed_tasks = _embedded_images_per_page(file_path, pages, empty_set)  # B 类：混合页嵌入图

    if not render_tasks and not embed_tasks:
        return still_failed  # 纯文本/纯空白：零模型调用，账目原样返回

    # ---------- 互斥序列第 1 步：若 LLM 驻留则先卸 ----------
    try:
        active = await gateway.active_models()
        if any(
            name == settings.llm_model or name.startswith(settings.llm_model + ":")
            for name in active
        ):
            await gateway.unload(settings.llm_model)
            logger.info("OCR 前已卸载 LLM: %s", settings.llm_model)
    except UpstreamError as e:
        logger.warning("查询/卸载 LLM 驻留状态失败（继续尝试 OCR）: %s", e.message)

    # ---------- 批次驻留策略：批内共享一次加载，批末兑现用完即卸 ----------
    ocr_keep = str(settings.ocr_keep_alive)
    hold = _BATCH_HOLD if ocr_keep == "0" else ocr_keep
    attempted = False

    async def _ocr_one(b64: str) -> str | None:
        """识别单张图；失败返回 None（单图故障不连坐其他图/页）。"""
        nonlocal attempted
        attempted = True  # 只要发起过请求，批末就要兜底卸载（模型可能已加载）
        try:
            out = await gateway.generate(
                model=settings.ocr_model,  # 模型名只从配置读（fallback 改 .env 一行即可）
                prompt=OCR_PROMPT,
                images=[b64],
                keep_alive=hold,  # 批内驻留共享一次加载；批末统一卸（见 docstring 红线第 2 条）
                num_predict=OCR_NUM_PREDICT,
                temperature=0.0,
            )
            return out.strip() or None
        except UpstreamError as e:
            logger.warning("OCR 单图失败: %s — %s", file_path.name, e.message)
            return None

    # ---------- A 类：空页整页识别 → 写回 ----------
    for pn, images in render_tasks:
        texts = [t for b64 in images if (t := await _ocr_one(b64))]
        if texts:
            pages_by_no[pn].text = "\n".join(texts)
        elif pn not in still_failed:
            # 识别失败：初始清单只收了无图页，这里补进「有图但识别失败」的页
            still_failed.append(pn)
    still_failed.sort()

    # ---------- B 类：混合页嵌入图识别 → 追加（失败只记日志，不动账） ----------
    for pn, images in embed_tasks.items():
        texts = [t for b64 in images if (t := await _ocr_one(b64))]
        if texts:
            pages_by_no[pn].text = (
                f"{pages_by_no[pn].text}\n{EMBEDDED_IMAGE_MARKER}\n" + "\n".join(texts)
            )

    # ---------- 互斥序列收尾：批末用完即卸（兑现 OCR_KEEP_ALIVE=0） ----------
    if attempted and ocr_keep == "0":
        try:
            await gateway.unload(settings.ocr_model)
        except UpstreamError as e:
            # 卸载失败不阻断入库（模型 5 分钟内也会自动过期），日志留痕
            logger.warning("批末卸载 OCR 模型失败（将自动过期）: %s", e.message)

    return still_failed


__all__ = [
    "ocr_fill_pages",
    "OCR_PROMPT",
    "OCR_NUM_PREDICT",
    "EMBEDDED_IMAGE_MARKER",
]
