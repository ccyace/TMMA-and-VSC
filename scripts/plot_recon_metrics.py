"""从 recon_metrics_log.json 重绘 BRECQ 曲线（可先手改 JSON 再运行）。"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qdiff.recon_metrics import replot_from_log


def main() -> None:
    p = argparse.ArgumentParser(description="重绘 BRECQ recon_metrics_log.json")
    p.add_argument(
        "--log",
        type=str,
        required=True,
        help="recon_metrics_log.json 路径",
    )
    p.add_argument("--out_dir", type=str, default="", help="图输出目录，默认同目录下 figures_replot")
    p.add_argument("--stages", type=str, default="", help="逗号分隔：weight,act；空=全部")
    p.add_argument("--layers", type=str, default="", help="逗号分隔层名；空=全部已记录层")
    args = p.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] or None
    layers = [s.strip() for s in args.layers.split(",") if s.strip()] or None
    replot_from_log(args.log, out_dir=args.out_dir, stages=stages, layers=layers)


if __name__ == "__main__":
    main()
