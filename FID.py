from __future__ import annotations

import argparse
import os

from pytorch_fid.fid_score import calculate_fid_from_gen_npz

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def calculate_fid(gen_npz: str) -> float:
    return calculate_fid_from_gen_npz(gen_npz)


def validate_image_dir(img_dir: str) -> bool:
    if not os.path.isdir(img_dir):
        print(f"目录不存在: {img_dir}")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description="从生成图 npz 计算 FID")
    p.add_argument(
        "gen",
        type=str,
        nargs="?",
        default=None,
        help="生成图 .npz（需含 mu/sigma）",
    )
    p.add_argument(
        "--gen",
        dest="gen_opt",
        type=str,
        default=None,
        help="同上，可用 --gen path.npz",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_npz = args.gen_opt or args.gen
    if not gen_npz:
        raise SystemExit("请指定生成图 npz，例如: python FID.py w8a8test3.70_fid_stats.npz")

    fid_value = calculate_fid(gen_npz)
    print(f"FID 值: {fid_value:.6f}")
