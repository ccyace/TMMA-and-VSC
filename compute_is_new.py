#!/usr/bin/env python3
"""Compute Inception Score with torch_fidelity (Q-Diffusion / paper-aligned).

Example:
  python compute_is_torch_fidelity.py --path output/w4a8_ourtwo_40step_50k
  python compute_is_torch_fidelity.py --path output/w4a8_ourtwo_50step_50k --batch_size 64
"""
from __future__ import annotations

import argparse
import os

from torch_fidelity import calculate_metrics


def default_inception_weights() -> str | None:
    home = os.path.expanduser("~")
    candidates = [
        "/root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth",
        "/root/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth",
        os.path.join(home, ".cache", "torch", "hub", "checkpoints", "weights-inception-2015-12-05-6726825d.pth"),
        os.path.join(home, ".cache", "torch", "hub", "checkpoints", "pt_inception-2015-12-05-6726825d.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="IS via torch_fidelity")
    parser.add_argument("--path", required=True, help="directory of generated png/jpg images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--weights",
        default=None,
        help="optional local inception weights; if omitted, try cache then download",
    )
    parser.add_argument("--out", default=None, help="optional txt to save the result")
    args = parser.parse_args()

    img_dir = os.path.abspath(args.path)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)

    weights = args.weights or default_inception_weights()
    kwargs = dict(
        input1=img_dir,
        isc=True,
        fid=False,
        kid=False,
        verbose=True,
        batch_size=args.batch_size,
        cuda=not args.cpu,
    )
    if weights:
        kwargs["feature_extractor_weights_path"] = weights
        print(f"using weights: {weights}", flush=True)
    else:
        print("no local weights found; torch_fidelity may download them", flush=True)

    metrics = calculate_metrics(**kwargs)
    mean = metrics["inception_score_mean"]
    std = metrics["inception_score_std"]
    line = f"IS mean={mean:.6f} std={std:.6f}"
    print(line, flush=True)
    print(metrics, flush=True)

    out = args.out
    if out is None:
        out = img_dir.rstrip("/") + "_is.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(str(metrics) + "\n")
        f.write(line + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
