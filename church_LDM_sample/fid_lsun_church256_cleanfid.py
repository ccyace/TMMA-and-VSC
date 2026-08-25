r"""
在「没有 LSUN Church 原始数据集」的情况下，对生成图计算 FID。

做法：使用 clean-fid 提供的 LSUN Church 256 预计算参考分布（Inception 特征上的 mu/sigma），
与你的生成图目录做 Fréchet 距离。参考统计文件首次运行时会自动下载到 cleanfid 包内 stats 目录。

若首次下载中断，缓存里可能留下损坏的 .npz，会触发 BadZipFile。本脚本会在运行前校验参考文件，
无效则自动删除并重新下载（也可用 --force_redownload_ref 强制重下）。

依赖：
    pip install clean-fid

示例：
    python church_LDM_sample\fid_lsun_church256_cleanfid.py --gen_dir "C:\Users\wsh\Desktop\q-diffusion-master\church_w4a8_3000\models\ldm\lsun_churches256\samples\2026-04-29-18-25-15\img"
    python fid_lsun_church256_cleanfid.py --gen_dir ".\w4a8" --force_redownload_ref
"""

from __future__ import annotations

import argparse
from typing import Optional
import os
import platform
import shutil
import sys
import urllib.request
from urllib.parse import urljoin


def _parse_args():
    p = argparse.ArgumentParser(
        description="生成图 vs LSUN Church 256（clean-fid 预计算参考）的 FID，无需本地真实数据集。"
    )
    p.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="生成图片所在目录（递归扫描 png/jpg/webp 等）",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="LSUN_church",
        help="clean-fid 参考集名称（默认 LSUN_church，对应 lsun_church_clean_train_256.npz）",
    )
    p.add_argument("--dataset_res", type=int, default=256, help="参考统计分辨率，church 一般为 256")
    p.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "test", "custom"],
        help="参考统计 split（LSUN church 预计算文件为 train）",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="clean",
        choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
        help="特征提取与参考统计的 mode，需与参考 npz 一致；LSUN church 官方提供的是 clean",
    )
    p.add_argument("--batch_size", type=int, default=64, help="特征提取 batch size")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers；默认 Windows 为 0，其它系统为 min(8, cpu)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cuda:0 / cpu；默认自动选择",
    )
    p.add_argument(
        "--use_dataparallel",
        action="store_true",
        help="对特征提取器使用 DataParallel（单卡一般不需要）",
    )
    p.add_argument(
        "--force_redownload_ref",
        action="store_true",
        help="强制删除 cleanfid 缓存中的参考 npz 并重新下载（修复损坏缓存）",
    )
    p.add_argument(
        "--ref_npz",
        type=str,
        default=None,
        help="手动指定已下载好的参考统计 .npz（需含 mu、sigma；将复制到 cleanfid stats 目录覆盖同名文件）",
    )
    return p.parse_args()


def _reference_rel_filename(
    dataset_name: str,
    mode: str,
    dataset_split: str,
    dataset_res: int,
    model_name: str = "inception_v3",
) -> str:
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    return (f"{dataset_name}_{mode}{model_modifier}_{dataset_split}_{dataset_res}.npz").lower()


def _npz_stats_valid(path: str) -> bool:
    import numpy as np

    if not os.path.isfile(path) or os.path.getsize(path) < 4096:
        return False
    try:
        with np.load(path) as z:
            if "mu" not in z.files or "sigma" not in z.files:
                return False
            if z["mu"].size == 0 or z["sigma"].size == 0:
                return False
    except Exception:
        return False
    return True


def _ensure_cleanfid_reference_npz(
    *,
    dataset_name: str,
    mode: str,
    dataset_split: str,
    dataset_res: int,
    model_name: str,
    force_redownload: bool,
    ref_npz: Optional[str],
) -> str:
    """返回 cleanfid 将使用的本地参考 npz 绝对路径。"""
    import cleanfid

    rel = _reference_rel_filename(
        dataset_name, mode, dataset_split, dataset_res, model_name=model_name
    )
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    local_path = os.path.join(stats_folder, rel)
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    url = urljoin(base_url, rel)

    if ref_npz:
        src = os.path.abspath(ref_npz)
        if not os.path.isfile(src):
            print(f"错误：--ref_npz 不存在: {src}", file=sys.stderr)
            sys.exit(1)
        if not _npz_stats_valid(src):
            print(f"错误：--ref_npz 不是有效的统计文件（需含 mu/sigma）: {src}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(stats_folder, exist_ok=True)
        shutil.copy2(src, local_path)
        print(f"已用本地文件覆盖参考统计: {local_path}")
        return local_path

    if force_redownload and os.path.isfile(local_path):
        os.remove(local_path)
        print(f"已按 --force_redownload_ref 删除: {local_path}")

    if _npz_stats_valid(local_path):
        return local_path

    if os.path.exists(local_path):
        print(f"参考统计无效或损坏，删除后重下: {local_path}")
        try:
            os.remove(local_path)
        except OSError as e:
            print(f"无法删除损坏文件: {e}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(stats_folder, exist_ok=True)
    part_path = local_path + ".part"
    print(f"正在下载参考统计 ({rel}) …")
    print(f"  URL: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; fid_lsun_church256_cleanfid/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(part_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(part_path, local_path)
    except BaseException:
        if os.path.isfile(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        raise

    if not _npz_stats_valid(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass
        print(
            "错误：下载完成但仍无法读取参考 npz。可换网络重试，或手动下载后使用:\n"
            f"  {url}\n"
            f"  保存为上述文件名后放入:\n"
            f"  {stats_folder}\n"
            "或使用: --ref_npz <你的文件路径>",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"参考统计已就绪: {local_path}")
    return local_path


def main():
    args = _parse_args()
    gen_dir = os.path.abspath(args.gen_dir)
    if not os.path.isdir(gen_dir):
        print(f"错误：生成图目录不存在: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from cleanfid import fid
    except ImportError as e:
        print("未安装 clean-fid 或依赖缺失。请先执行：", file=sys.stderr)
        print("  pip install clean-fid", file=sys.stderr)
        print(f"原始错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_workers is None:
        if platform.system() == "Windows":
            num_workers = 0
        else:
            num_workers = min(8, os.cpu_count() or 1)
    else:
        num_workers = args.num_workers

    _ensure_cleanfid_reference_npz(
        dataset_name=args.dataset_name,
        mode=args.mode,
        dataset_split=args.dataset_split,
        dataset_res=args.dataset_res,
        model_name="inception_v3",
        force_redownload=args.force_redownload_ref,
        ref_npz=args.ref_npz,
    )

    print(f"gen_dir      = {gen_dir}")
    print(
        f"reference    = clean-fid: {args.dataset_name} / res={args.dataset_res} / "
        f"split={args.dataset_split} / mode={args.mode}"
    )
    print(f"device       = {device}")
    print(f"batch_size   = {args.batch_size}")
    print(f"num_workers  = {num_workers}")

    score = fid.compute_fid(
        fdir1=gen_dir,
        fdir2=None,
        mode=args.mode,
        model_name="inception_v3",
        num_workers=num_workers,
        batch_size=args.batch_size,
        device=device,
        dataset_name=args.dataset_name,
        dataset_res=args.dataset_res,
        dataset_split=args.dataset_split,
        verbose=True,
        use_dataparallel=args.use_dataparallel,
    )
    print(
        f"\nFID (clean-fid vs {args.dataset_name} {args.dataset_res}px, "
        f"split={args.dataset_split}): {score:.4f}"
    )


if __name__ == "__main__":
    main()
