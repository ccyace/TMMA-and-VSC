r"""
计算文件夹内生成图的 Inception Score (IS)。

使用 torchvision Inception v3 (ImageNet 预训练)，对每张图做 softmax 后按标准 IS 公式：
在若干 split 上估计 exp(E_x[ KL(p(y|x) || \bar p(y)) ]) 的均值与标准差。

依赖：torch, torchvision, pillow, numpy, scipy, tqdm

示例：
    python church_LDM_sample/is_inception_score.py --img_dir "C:\\Users\\...\\add_w8a8_9.81"
    python church_LDM_sample/is_inception_score.py --img_dir ./add_w8a8_9.81 --batch_size 32 --splits 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.stats import entropy
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
from tqdm import tqdm


def _collect_images(root: Path, recursive: bool) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if recursive:
        paths = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    else:
        paths = [
            root / n
            for n in os.listdir(root)
            if (root / n).is_file() and (root / n).suffix.lower() in exts
        ]
    return sorted(paths, key=lambda p: p.as_posix().lower())


class ImageListDataset(Dataset):
    def __init__(self, paths: list[Path], transform: T.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def inception_score_from_preds(preds: np.ndarray, splits: int) -> tuple[float, float]:
    """preds: (N, 1000) softmax 概率。"""
    n = preds.shape[0]
    if n < splits:
        raise ValueError(f"图片数 {n} 小于 splits={splits}，请减小 --splits 或增加图片。")
    split_size = n // splits
    scores = []
    for k in range(splits):
        part = preds[k * split_size : (k + 1) * split_size, :]
        py = np.mean(part, axis=0)
        # 每列：样本 i 的 p(y|x_i) 与边际 py 的 KL；对样本维求平均后取 exp
        kl = entropy(part.T, py[:, np.newaxis], axis=0)
        scores.append(float(np.exp(np.mean(kl))))
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def run_is(
    img_dir: Path,
    batch_size: int,
    splits: int,
    device: torch.device,
    num_workers: int,
    recursive: bool,
) -> tuple[float, float]:
    paths = _collect_images(img_dir, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"目录下未找到图片: {img_dir}")

    # Inception v3 ImageNet 预处理（与 torchvision 预训练权重一致）
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = ImageListDataset(paths, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    model = model.to(device)
    model.eval()

    preds_list = []
    for batch in tqdm(loader, desc="Inception Score"):
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        prob = torch.nn.functional.softmax(logits, dim=1)
        preds_list.append(prob.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    return inception_score_from_preds(preds, splits=splits)


def main() -> None:
    p = argparse.ArgumentParser(description="文件夹图像 Inception Score (IS)")
    p.add_argument(
        "--img_dir",
        type=str,
        default=r"C:\Users\wsh\Desktop\q-diffusion-master\church_LDM_sample\w4a8_2000_9.86\img",
        help="图片目录",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--splits", type=int, default=10, help="IS 估计的 split 数（需整除样本数或会丢弃余数）")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归子目录扫描图片（默认仅当前目录）",
    )
    args = p.parse_args()

    img_dir = Path(args.img_dir).resolve()
    if not img_dir.is_dir():
        print(f"错误：目录不存在 {img_dir}", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nw = args.num_workers
    if nw is None:
        nw = 0 if os.name == "nt" else min(8, os.cpu_count() or 1)

    print(f"img_dir      = {img_dir}")
    print(f"device       = {device}")
    print(f"batch_size   = {args.batch_size}")
    print(f"splits       = {args.splits}")
    print(f"recursive    = {args.recursive}")

    mean_is, std_is = run_is(
        img_dir,
        batch_size=args.batch_size,
        splits=args.splits,
        device=device,
        num_workers=nw,
        recursive=args.recursive,
    )
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")


if __name__ == "__main__":
    main()
