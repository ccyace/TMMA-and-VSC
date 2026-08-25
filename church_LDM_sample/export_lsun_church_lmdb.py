r"""
从 LSUN Church Outdoor **训练集 LMDB** 中导出图片（LMDB 里存的是 WebP 字节流）。

你在网页上下载的 `church_outdoor_train_lmdb.zip` 解压后应得到类似目录：
  church_outdoor_train_lmdb/
    data.mdb
    lock.mdb

本脚本把条目按顺序导出为 `.webp`（与官方 fyu/lsun 的 `data.py export` 一致），
可用 `--max_images` 只导出前 N 张，省磁盘与时间。FID / sFID 参考图可用 webp
（clean-fid 与仓库内 sfid 脚本均支持常见图片格式）。

依赖：pip install lmdb

官方仓库（含交互查看等）：https://github.com/fyu/lsun

示例：
  python church_LDM_sample/export_lsun_church_lmdb.py ^
    --lmdb_dir "D:\data\church_outdoor_train_lmdb" ^
    --out_dir "D:\data\lsun_church_train_webp" ^
    --max_images 10000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def export_flat(
    lmdb_dir: str,
    out_dir: str,
    max_images: Optional[int] = None,
) -> int:
    import lmdb

    db = Path(lmdb_dir)
    if not db.is_dir():
        raise FileNotFoundError(f"LMDB 目录不存在: {db}")
    if not (db / "data.mdb").is_file():
        raise FileNotFoundError(f"未找到 data.mdb，请确认 --lmdb_dir 指向解压后的 LMDB 文件夹: {db}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(db),
        map_size=1099511627776,
        max_readers=100,
        readonly=True,
        lock=False,
    )
    count = 0
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for key, val in cur:
            try:
                name = key.decode("ascii")
            except UnicodeDecodeError:
                name = key.hex()
            # 与官方 data.py 一致：直接写入 webp 字节
            dst = out / f"{name}.webp"
            dst.write_bytes(val)
            count += 1
            if max_images is not None and count >= max_images:
                break
            if count % 2000 == 0:
                print(f"  已导出 {count} 张 …")
    env.close()
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="从 LSUN Church LMDB 导出 webp 到文件夹（扁平结构）")
    p.add_argument(
        "--lmdb_dir",
        type=str,
        required=True,
        help="解压后含 data.mdb / lock.mdb 的目录路径",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录（全部 .webp 平铺在此目录下）",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="最多导出多少张；不指定则导出整个数据库（体积很大）",
    )
    args = p.parse_args()

    try:
        import lmdb  # noqa: F401
    except ImportError:
        print("请先安装: pip install lmdb", file=sys.stderr)
        sys.exit(1)

    n = export_flat(args.lmdb_dir, args.out_dir, max_images=args.max_images)
    print(f"完成，共导出 {n} 张 -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
