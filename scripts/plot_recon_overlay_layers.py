"""
每层一张图：该层 Baseline vs TMMA 的 loss / grad 叠在一起（颜色区分方法）。

默认生成：
  - down.0.block.0  → comparison_figs/down_0_block_0_baseline_vs_tmma.png
  - down.1.attn.0   → comparison_figs/down_1_attn_0_baseline_vs_tmma.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdiff.recon_metrics import normalize_layer_name, load_layer_whitelist, DEFAULT_LAYERS_FILE

METHOD_STYLE = {
    "baseline": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8, "label": "Baseline"},
    "tmma": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.8, "label": "TMMA"},
}


def load_series(log_path: str, stage: str, layer: str) -> List[Dict]:
    layer = normalize_layer_name(layer)
    with open(log_path, encoding="utf-8") as f:
        blob = json.load(f)
    entry = blob.get("stages", {}).get(stage, {}).get(layer)
    if not entry:
        raise KeyError(f"{log_path} 中无 stage={stage!r} layer={layer!r}")
    series = entry.get("series", entry) if isinstance(entry, dict) else entry
    return series


def plot_one_layer(
    layer: str,
    baseline_log: str,
    baseline_stage: str,
    tmma_log: str,
    tmma_stage: str,
    out_path: str,
    tmma_label: str = "TMMA",
) -> None:
    layer = normalize_layer_name(layer)
    short = layer.replace("model.", "").replace(".", "_")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for key, log_path, stage in [
        ("baseline", baseline_log, baseline_stage),
        ("tmma", tmma_log, tmma_stage),
    ]:
        try:
            series = load_series(log_path, stage, layer)
        except KeyError as e:
            print(f"[skip] {layer} {key}: {e}")
            continue
        sty = METHOD_STYLE[key]
        disp = sty["label"] if key == "baseline" else tmma_label
        iters = [r["iter"] for r in series]
        axes[0].plot(
            iters,
            [r["rec"] for r in series],
            color=sty["color"],
            linestyle=sty["linestyle"],
            linewidth=sty["linewidth"],
            label=disp,
            alpha=0.92,
        )
        axes[1].plot(
            iters,
            [r["grad_norm"] for r in series],
            color=sty["color"],
            linestyle=sty["linestyle"],
            linewidth=sty["linewidth"],
            label=disp,
            alpha=0.92,
        )

    axes[0].set_ylabel("reconstruction loss (rec)")
    axes[0].set_title(f"{layer} — Baseline vs TMMA")
    axes[0].legend(loc="upper right", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("grad L2 norm")
    axes[1].legend(loc="upper right", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="每层单独：Baseline vs TMMA 叠图")
    p.add_argument(
        "--baseline_log",
        type=str,
        default=str(_REPO_ROOT / "baseline_recon_curves_run/recon_curves/recon_metrics_log.json"),
    )
    p.add_argument(
        "--tmma_log",
        type=str,
        default=str(_REPO_ROOT / "tmma_recon_curves_run/recon_curves/recon_metrics_log.json"),
    )
    p.add_argument("--baseline_stage", type=str, default="act")
    p.add_argument("--tmma_stage", type=str, default="act_g0")
    p.add_argument(
        "--layers",
        type=str,
        default="",
        help="逗号分隔层名；空则读取 configs/recon_key_layers.json 全部层",
    )
    p.add_argument(
        "--layers_file",
        type=str,
        default="",
        help="层列表 JSON，默认 configs/recon_key_layers.json",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_REPO_ROOT / "comparison_figs"),
    )
    p.add_argument("--tmma_label", type=str, default="", help="图例 TMMA 名称，默认 TMMA (stage)")
    args = p.parse_args()

    if args.layers.strip():
        layers = [normalize_layer_name(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers_file = args.layers_file.strip() or str(DEFAULT_LAYERS_FILE)
        layers = load_layer_whitelist(layers_file=layers_file)
    print(f"共 {len(layers)} 层 -> {args.out_dir}")
    tmma_label = args.tmma_label or f"TMMA ({args.tmma_stage})"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        short = layer.replace("model.", "").replace(".", "_")
        out_path = out_dir / f"{short}_baseline_vs_tmma.png"
        plot_one_layer(
            layer,
            args.baseline_log,
            args.baseline_stage,
            args.tmma_log,
            args.tmma_stage,
            str(out_path),
            tmma_label=tmma_label,
        )


if __name__ == "__main__":
    main()
