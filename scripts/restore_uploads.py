# -*- coding: utf-8 -*-
"""
上传副本恢复脚本：源文件在、副本丢 → 按文件名从原始课件库批量拷回。

为什么需要它（2026-09-25 事件）：data/uploads 里 37 个 PDF 被某不明程序
批量永久删除（回收站为空、Defender 无隔离记录、DB 行与向量都完好）——
检索问答不受影响，但「查看原页」与增量重传依赖源文件。原始课件在本机
（如 E:\\AsusDownload），按文件名精确匹配拷回即可，零重算、零重新上传。

用法：
    python scripts/restore_uploads.py                     # 默认扫 E:/AsusDownload
    python scripts/restore_uploads.py --source "D:/课件"   # 指定原始课件库根目录
    python scripts/restore_uploads.py --dry-run            # 只报告将恢复什么，不动文件
退出码：全部缺失文件都找回 → 0；有未匹配到的 → 1（把文件名清单打给你去补源）。
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# 直接 `python scripts/xxx.py` 运行时 sys.path[0]=scripts/，import app 会失败——
# 先把项目根塞进搜索路径（demo_e2e 不依赖 app 所以从没暴露这个问题）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal  # noqa: E402
from app.db import crud  # noqa: E402

# 扫描超时（秒）：原始课件库可能挂在大容量盘，限时防止在全盘爬虫里卡死
SCAN_BUDGET_SECONDS = 300


def build_index(source_root: Path) -> dict[str, list[Path]]:
    """扫原始课件库 → {文件名: [候选路径]}（同名多份全部记录，取第一份）。"""
    index: dict[str, list[Path]] = {}
    t0 = time.time()
    for dirpath, _dirnames, filenames in os.walk(source_root):
        if time.time() - t0 > SCAN_BUDGET_SECONDS:
            print(f"[restore] 扫描超过 {SCAN_BUDGET_SECONDS}s 提前停止（已索引 {sum(map(len, index.values()))} 个文件）")
            break
        for name in filenames:
            index.setdefault(name, []).append(Path(dirpath) / name)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="从原始课件库恢复 data/uploads 缺失副本")
    parser.add_argument("--source", default="E:/AsusDownload", help="原始课件库根目录")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不拷贝")
    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.is_dir():
        print(f"[restore] 源目录不存在: {source_root}（用 --source 指定课件原位置）")
        return 1

    print(f"[restore] 索引源目录: {source_root}")
    index = build_index(source_root)
    print(f"[restore] 已索引 {sum(map(len, index.values()))} 个文件")

    restored: list[str] = []
    missing: list[str] = []
    db = SessionLocal()
    try:
        for group in crud.list_kb_groups(db, user_id=1):  # TODO 多账号时按 --user 扩展
            for doc in crud.list_documents(db, user_id=1, group_id=group.id):
                dst = Path(doc.file_path)
                if dst.is_file():
                    continue  # 副本还在，无需恢复
                candidates = index.get(doc.file_name, [])
                if not candidates:
                    missing.append(doc.file_name)
                    continue
                if args.dry_run:
                    restored.append(doc.file_name)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(candidates[0], dst)  # copy2：连修改时间一起保留，便于对账
                    restored.append(doc.file_name)
                except OSError as e:
                    missing.append(f"{doc.file_name}（拷贝失败: {e}）")
    finally:
        db.close()

    for name in restored:
        print(f"  OK   {name}")
    for name in missing:
        print(f"  MISS {name}")
    print(f"[restore] 恢复 {len(restored)} 个，未找回 {len(missing)} 个")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
