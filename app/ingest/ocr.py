# -*- coding: utf-8 -*-
"""
OCR 回填层（M3 多模态入库核心）：无文本层的页经 GLM-OCR 转文本，写回 ParsedPage。

★ 显存互斥序列（CLAUDE.md §3 红线在本阶段的代码落点，答辩重点讲这里）：
1. 先查 /api/ps 确认 LLM 是否驻留——驻留才卸载（对未加载模型发 keep_alive=0
   会让 Ollama 先加载再卸载，白等几十秒）；
2. GLM-OCR 调用显式 keep_alive=settings.ocr_keep_alive（=0）——用完即卸，显存立刻归还；
3. 全程经 gateway（排队闸门 + 互斥锁），与任何生成调用天然串行，绝无并发推理；
4. OCR 全部结束后流水线才开始向量化——时序上「OCR 与 LLM 时段隔离」自动成立。

为什么 OCR 失败不炸整个入库：文本层内容仍有价值（一页识别失败不该连坐全部已解析内容），
失败页码保留在 empty_pages 返回给调用方——用户重新上传即可重试，也给后续调优留了账。

页图像来源按格式分发（与 parsers 的「格式差异收口」同一思路）：
- 图片文件：文件字节本身就是图；
- PDF：pymupdf 把该页渲染成 PNG（pypdf 只能读文本层，渲染必须靠 pymupdf——见 requirements 注释）；
- PPT：抽取该幻灯片内嵌图片的原始字节（blob），一图一调、结果拼接；
- Word：无「页的图像」概念（逻辑页），若 Word 出现空页保持登记即可。
"""
import base64
import logging
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import UpstreamError
from app.core.llm import gateway
from app.ingest.parsers import ParsedPage

logger = logging.getLogger(__name__)

# OCR 任务提示：GLM-OCR 是「图 → Markdown」专职模型，指令要求逐字转换不加解释
OCR_PROMPT = "请将图像中的全部文字、公式与表格按原文顺序转换为 Markdown 文本，不要添加任何解释。"

# 整页 OCR 的输出上限：比生成类默认 512 大（一页文档轻松过 500 token），
# 但仍受控（红线要求输出长度有界）；temperature=0 保证识别确定性
OCR_NUM_PREDICT = 2048


def _to_b64(data: bytes) -> str:
    """字节流转 base64 ASCII 串（Ollama images 参数格式，不带 data: 前缀）。"""
    return base64.b64encode(data).decode("ascii")


def _page_images_b64(file_path: Path, page_no: int) -> list[str]:
    """取某页对应的图像（0..n 张，base64）。没有图像来源（如空白页）返回 []。"""
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".png", ".jpg", ".jpeg"}:
            # 图片文件：自身即唯一图像来源
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
            # PPT：取该幻灯片内嵌图片的原始字节（多图全部返回，逐图识别后拼接）
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE

            prs = Presentation(str(file_path))
            if page_no - 1 >= len(prs.slides):
                return []
            images: list[str] = []
            for shape in prs.slides[page_no - 1].shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    images.append(_to_b64(shape.image.blob))
            return images
    except Exception:
        # 取图失败（文件损坏等）视作「无图像来源」→ 该页保留在 empty_pages
        logger.exception("取页图像失败: %s 第%s页", file_path.name, page_no)
        return []

    return []  # Word 等无页图像格式


async def ocr_fill_pages(
    file_path: Path, pages: list[ParsedPage], empty_pages: list[int]
) -> list[int]:
    """对 empty_pages 逐页 OCR 并把文本写回对应 ParsedPage，返回「仍失败」的页码。

    返回值直接作为最终的 empty_pages 落库——成功回填的页移出清单（进向量库），
    失败的页留在清单（前端提示、可重传重试）。
    """
    if not empty_pages:
        return []  # 纯文本文档不碰 OCR 路径：不查驻留、不卸 LLM、零模型调用

    # --- 互斥序列第 1 步：若 LLM 驻留则先卸（查驻留失败不阻断——OCR 自身失败会走容错） ---
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

    still_failed: list[int] = []
    for page_no in empty_pages:
        images = _page_images_b64(file_path, page_no)
        if not images:
            # 无图像来源（纯空白页/Word 逻辑页）：保持登记，不白跑 OCR
            still_failed.append(page_no)
            continue

        texts: list[str] = []
        for b64 in images:
            try:
                # --- 互斥序列第 2 步：keep_alive=0 用完即卸（红线） ---
                out = await gateway.generate(
                    model=settings.ocr_model,  # 模型名只从配置读（fallback 改 .env 一行即可）
                    prompt=OCR_PROMPT,
                    images=[b64],
                    keep_alive=settings.ocr_keep_alive,  # "0" = 识别完立刻从显存踢掉
                    num_predict=OCR_NUM_PREDICT,
                    temperature=0.0,
                )
                if out.strip():
                    texts.append(out.strip())
            except UpstreamError as e:
                logger.warning(
                    "OCR 失败（该页保留 empty_pages）: %s 第%s页 — %s",
                    file_path.name, page_no, e.message,
                )
                break  # 本页放弃（多图页半张拼接无意义），下一页继续尝试

        if texts:
            for page in pages:
                if page.page_no == page_no:
                    page.text = "\n".join(texts)
                    break
        else:
            still_failed.append(page_no)

    return still_failed


__all__ = ["ocr_fill_pages", "OCR_PROMPT", "OCR_NUM_PREDICT"]
