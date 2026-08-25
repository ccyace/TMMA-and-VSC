r"""
在「没有 LSUN Bedroom 原始数据集」的情况下，对生成图计算 FID。

参考统计（二选一，均无需本地 LSUN 原图）：
  1. TTUR 官方 fid_stats_lsun_train.npz（与 NCSNv2 / DDIM 论文最可比，约 33MB）
  2. OpenAI guided-diffusion 的 VIRTUAL_lsun_bedroom256.npz（约 1GB，国内网络通常可下）
     用 pytorch_fid 从中提取 50k 参考图的 mu/sigma 并缓存（--ref_source openai / auto 回退）

依赖：
    pip install pytorch-fid pillow torch numpy

示例：
    python fid_lsun_bedroom256_pytorchfid.py

    # TTUR 被墙时用 OpenAI 参考包（首次会下载 ~1GB 并缓存统计）
    python fid_lsun_bedroom256_pytorchfid.py --ref_source openai

    # 自动：先 TTUR，失败则 OpenAI
    python fid_lsun_bedroom256_pytorchfid.py --ref_source auto

    # 已有手动下载的 TTUR npz
    python fid_lsun_bedroom256_pytorchfid.py --ref_npz stats\fid_stats_lsun_train.npz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# TTUR 官方（国内常连不上 bioinf.jku.at → WinError 10061）
TTUR_REF_URLS = [
    "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz",
    "https://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz",
]
OPENAI_VIRTUAL_REF_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/"
    "lsun/bedroom/VIRTUAL_lsun_bedroom256.npz"
)

DEFAULT_TTUR_FILENAME = "fid_stats_lsun_train.npz"
DEFAULT_OPENAI_RAW_FILENAME = "VIRTUAL_lsun_bedroom256.npz"
DEFAULT_OPENAI_STATS_FILENAME = "fid_stats_lsun_openai_virtual50k_pytorch.npz"
DEFAULT_STATS_DIR = os.path.join(_REPO_ROOT, "stats")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
USER_AGENT = "Mozilla/5.0 (compatible; fid_lsun_bedroom256_pytorchfid/1.0)"


def validate_image_dir(img_dir: str) -> bool:
    """检查目录下图像能否正常读取（逻辑与 FID.py 一致）。"""
    from PIL import Image

    bad_files = []
    for root, _, files in os.walk(img_dir):
        for name in files:
            path = os.path.join(root, name)
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                bad_files.append(path)

    if bad_files:
        print("发现无法识别或损坏的文件：")
        for f in bad_files:
            print("  ", f)
        print("请删除或重新生成上述文件后再计算 FID。")
        return False
    return True


def _parse_args():
    p = argparse.ArgumentParser(
        description="生成图 vs LSUN Bedroom 256 的 FID（pytorch_fid），无需本地真实数据集。"
    )
    p.add_argument(
        "--gen_dir",
        type=str,
        default=os.path.join(
            _REPO_ROOT,
            "bedroom_w8a8",
            "samples",
            "2026-06-09-14-01-53",
            "img",
        ),
        help="生成图片目录",
    )
    p.add_argument(
        "--ref_npz",
        type=str,
        default=None,
        help="手动指定含 mu/sigma 的参考 .npz（指定后忽略 --ref_source 下载逻辑）",
    )
    p.add_argument(
        "--ref_source",
        type=str,
        default="auto",
        choices=["auto", "ttur", "openai"],
        help="参考统计来源：ttur=官方预计算；openai=guided-diffusion 50k 参考包；auto=先 ttur 后 openai",
    )
    p.add_argument(
        "--stats_dir",
        type=str,
        default=DEFAULT_STATS_DIR,
        help="缓存目录（默认 <repo>/stats）",
    )
    p.add_argument(
        "--force_redownload_ref",
        action="store_true",
        help="强制重新下载参考文件（TTUR npz 或 OpenAI 原始 npz）",
    )
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dims", type=int, default=2048)
    p.add_argument("--warn_below", type=int, default=50000)
    p.add_argument("--skip_validate", action="store_true")
    return p.parse_args()


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


def _download_file(url: str, dest_path: str, timeout: int = 3600) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    part_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part_path, "wb") as out:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        chunk = 1024 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total and downloaded % (50 * chunk) < chunk:
                pct = 100.0 * downloaded / total
                print(f"  已下载 {downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB ({pct:.1f}%)")
    os.replace(part_path, dest_path)


def _download_with_retries(urls: List[str], dest_path: str, retries: int = 2) -> str:
    errors = []
    for url in urls:
        for attempt in range(retries + 1):
            print(f"正在下载 … (尝试 {attempt + 1}/{retries + 1})")
            print(f"  URL: {url}")
            try:
                _download_file(url, dest_path)
                return url
            except Exception as e:
                errors.append(f"{url} -> {e}")
                if os.path.isfile(dest_path + ".part"):
                    try:
                        os.remove(dest_path + ".part")
                    except OSError:
                        pass
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("全部下载源均失败:\n  " + "\n  ".join(errors))


def ensure_ttur_reference_npz(stats_dir: str, force_redownload: bool) -> str:
    local_path = os.path.join(stats_dir, DEFAULT_TTUR_FILENAME)
    if force_redownload and os.path.isfile(local_path):
        os.remove(local_path)
        print(f"已删除: {local_path}")

    if _npz_stats_valid(local_path):
        return local_path

    if os.path.exists(local_path):
        os.remove(local_path)

    _download_with_retries(TTUR_REF_URLS, local_path)
    if not _npz_stats_valid(local_path):
        os.remove(local_path)
        raise RuntimeError("TTUR 参考 npz 下载后校验失败")
    print(f"TTUR 参考统计已就绪: {local_path}")
    return local_path


def _load_openai_virtual_images_mmap(npz_path: str):
    """加载 OpenAI VIRTUAL_lsun_bedroom256.npz 中的 arr_0（NHWC uint8）。"""
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as z:
        if "arr_0" not in z.files:
            raise ValueError(f"{npz_path} 中无 arr_0 键，keys={z.files}")
        arr = z["arr_0"]
        if arr.ndim != 4:
            raise ValueError(f"arr_0 形状异常: {arr.shape}，期望 [N,H,W,C]")
    # 重新 mmap 打开，避免整包进内存
    return np.load(npz_path, mmap_mode="r")["arr_0"]


def _compute_mu_sigma_from_nhwc(
    images_nhwc,
    device: str,
    batch_size: int,
    dims: int = 2048,
) -> Tuple[object, object]:
    """对 NHWC uint8 图像 batch 提取 Inception 特征并算 mu/sigma（与 pytorch_fid 一致）。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    n = int(images_nhwc.shape[0])
    pred_chunks = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_np = np.array(images_nhwc[i : i + batch_size], dtype=np.float32) / 255.0
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_chunks.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  特征提取进度: {min(i + batch_size, n)} / {n}")

    act = np.concatenate(pred_chunks, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def ensure_openai_reference_npz(
    stats_dir: str,
    force_redownload: bool,
    device: str,
    batch_size: int,
    dims: int,
) -> str:
    """
    下载 OpenAI VIRTUAL_lsun_bedroom256.npz，用 pytorch_fid Inception 算 mu/sigma 并缓存。
    与 TTUR 全训练集统计略有差异，但可用于国内网络下的相对对比。
    """
    import numpy as np

    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, DEFAULT_OPENAI_STATS_FILENAME)
    raw_path = os.path.join(stats_dir, DEFAULT_OPENAI_RAW_FILENAME)

    if force_redownload:
        for p in (stats_path, raw_path):
            if os.path.isfile(p):
                os.remove(p)
                print(f"已删除: {p}")

    if _npz_stats_valid(stats_path):
        print(f"使用已缓存的 OpenAI 参考统计: {stats_path}")
        return stats_path

    if not os.path.isfile(raw_path):
        print("正在下载 OpenAI guided-diffusion LSUN Bedroom 参考包（约 1GB，仅需一次）…")
        _download_with_retries([OPENAI_VIRTUAL_REF_URL], raw_path)

    print("从 OpenAI 参考包计算 pytorch_fid 兼容的 mu/sigma（约需数分钟）…")
    images = _load_openai_virtual_images_mmap(raw_path)
    print(f"  参考图像: shape={images.shape}, dtype={images.dtype}")
    mu, sigma = _compute_mu_sigma_from_nhwc(images, device, batch_size, dims)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma)
    print(f"OpenAI 参考统计已缓存: {stats_path}")
    return stats_path


def ensure_reference_npz(
    *,
    stats_dir: str,
    ref_npz: Optional[str],
    ref_source: str,
    force_redownload: bool,
    device: str,
    batch_size: int,
    dims: int,
) -> Tuple[str, str]:
    """返回 (npz_path, ref_description)。"""
    if ref_npz:
        path = os.path.abspath(ref_npz)
        if not os.path.isfile(path):
            print(f"错误：--ref_npz 不存在: {path}", file=sys.stderr)
            sys.exit(1)
        if not _npz_stats_valid(path):
            print(f"错误：--ref_npz 无效（需含 mu/sigma）: {path}", file=sys.stderr)
            sys.exit(1)
        return path, "user-specified npz"

    os.makedirs(stats_dir, exist_ok=True)

    if ref_source == "ttur":
        return ensure_ttur_reference_npz(stats_dir, force_redownload), "TTUR LSUN Bedroom train"

    if ref_source == "openai":
        path = ensure_openai_reference_npz(
            stats_dir, force_redownload, device, batch_size, dims
        )
        return path, "OpenAI guided-diffusion 50k reference (pytorch_fid stats)"

    # auto: TTUR → OpenAI
    try:
        path = ensure_ttur_reference_npz(stats_dir, force_redownload)
        return path, "TTUR LSUN Bedroom train"
    except Exception as e:
        print(
            "\nTTUR 官方源不可用（国内常见：bioinf.jku.at 连接被拒绝 WinError 10061）。"
            f"\n  原因: {e}"
            "\n  正在自动改用 OpenAI Azure 参考包 …\n"
        )
        path = ensure_openai_reference_npz(
            stats_dir, force_redownload, device, batch_size, dims
        )
        return path, "OpenAI guided-diffusion 50k reference (pytorch_fid stats, auto fallback)"


def count_images(root: str) -> int:
    n = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMAGE_EXTS):
                n += 1
    return n


def _print_manual_ttur_help():
    print(
        "\n若需与论文完全一致的 TTUR 参考，可手动下载后指定 --ref_npz：\n"
        "  http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz\n"
        "  保存为 stats/fid_stats_lsun_train.npz\n"
        "  （需代理/VPN 访问 JKU 站点）\n"
        "或直接使用: --ref_source openai\n"
    )


def main():
    args = _parse_args()
    gen_dir = os.path.abspath(args.gen_dir)
    if not os.path.isdir(gen_dir):
        print(f"错误：生成图目录不存在: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_validate and not validate_image_dir(gen_dir):
        print("生成图目录校验失败，已停止 FID。", file=sys.stderr)
        sys.exit(1)

    n_images = count_images(gen_dir)
    if n_images == 0:
        print(f"错误：目录下未找到图片: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from pytorch_fid import fid_score
    except ImportError as e:
        print("未安装 pytorch-fid。请执行: pip install pytorch-fid", file=sys.stderr)
        print(f"原始错误: {e}", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stats_dir = os.path.abspath(args.stats_dir)

    try:
        ref_path, ref_desc = ensure_reference_npz(
            stats_dir=stats_dir,
            ref_npz=args.ref_npz,
            ref_source=args.ref_source,
            force_redownload=args.force_redownload_ref,
            device=device,
            batch_size=args.batch_size,
            dims=args.dims,
        )
    except Exception as e:
        print(f"错误：无法准备参考统计: {e}", file=sys.stderr)
        _print_manual_ttur_help()
        sys.exit(1)

    print(f"gen_dir      = {gen_dir}")
    print(f"num_images   = {n_images}")
    print(f"reference    = {ref_desc}")
    print(f"ref_path     = {ref_path}")
    print(f"device       = {device}")
    print(f"batch_size   = {args.batch_size}")

    if args.warn_below > 0 and n_images < args.warn_below:
        print(
            f"\n警告：当前仅 {n_images} 张生成图，论文常用 50000 张；"
            "样本过少时 FID 不宜与论文直接对比。\n"
        )

    if "OpenAI" in ref_desc or "openai" in ref_path.lower():
        print(
            "提示：当前使用 OpenAI 50k 参考统计，与 TTUR 全训练集 FID 可能有偏差；"
            "模型间相对比较仍有效。\n"
        )

    fid_value = fid_score.calculate_fid_given_paths(
        [gen_dir, ref_path],
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
    )
    print(f"\nFID (pytorch_fid vs {ref_desc}): {fid_value:.4f}")


if __name__ == "__main__":
    main()
