"""BRECQ / AdaRound 重建：关键层 loss、grad 记录与绘图。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LAYERS_FILE = os.path.join(_REPO_ROOT, "configs", "recon_key_layers.json")
MASTER_LOG_NAME = "recon_metrics_log.json"


def normalize_layer_name(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    if name.startswith("model."):
        return name
    return f"model.{name}"


def load_layer_whitelist(
    layers_csv: str = "",
    layers_file: str = "",
) -> List[str]:
    """从逗号列表或 JSON 文件加载关键层名。"""
    names: List[str] = []
    if layers_file and os.path.isfile(layers_file):
        with open(layers_file, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, dict):
            raw = blob.get("layers", [])
        elif isinstance(blob, list):
            raw = blob
        else:
            raise ValueError(f"层列表文件格式无效: {layers_file}")
        names.extend(str(x) for x in raw)
    if layers_csv.strip():
        names.extend(x.strip() for x in layers_csv.split(",") if x.strip())
    if not names and os.path.isfile(DEFAULT_LAYERS_FILE):
        return load_layer_whitelist(layers_file=DEFAULT_LAYERS_FILE)
    seen: Set[str] = set()
    out: List[str] = []
    for n in names:
        key = normalize_layer_name(n)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def layer_in_whitelist(full_name: str, whitelist: Set[str]) -> bool:
    if not whitelist:
        return False
    return normalize_layer_name(full_name) in whitelist


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.replace(".", "_"))


def _plot_layer_curves(layer_name: str, records: List[Dict], png_path: str) -> None:
    if not records:
        return
    iters = [r["iter"] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(iters, [r["total"] for r in records], label="total", linewidth=1.2)
    axes[0].plot(iters, [r["rec"] for r in records], label="rec (MSE)", linewidth=1.0, alpha=0.85)
    axes[0].plot(iters, [r["round"] for r in records], label="round", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"BRECQ reconstruction — {layer_name}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, [r["grad_norm"] for r in records], color="#d62728", linewidth=1.2)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("grad L2 norm")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


class ReconMetricsRegistry:
    """汇总关键层指标到单一 JSON，并按层出图。"""

    def __init__(
        self,
        log_dir: str,
        stage: str,
        layer_whitelist: List[str],
        log_every: int = 10,
        max_iters: Optional[int] = 2000,
        layers_file: str = "",
    ):
        self.log_dir = os.path.abspath(log_dir)
        self.stage = stage
        self.log_every = max(1, int(log_every))
        self.max_iters = max_iters
        self.layers_file = os.path.abspath(layers_file) if layers_file else ""
        self.whitelist: Set[str] = {normalize_layer_name(x) for x in layer_whitelist}
        self.master_path = os.path.join(self.log_dir, MASTER_LOG_NAME)
        self.figures_dir = os.path.join(self.log_dir, "figures", stage)
        os.makedirs(self.figures_dir, exist_ok=True)
        self._blob = self._load_or_init()

    def _load_or_init(self) -> Dict:
        if os.path.isfile(self.master_path):
            with open(self.master_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if isinstance(blob, dict) and "stages" in blob:
                return blob
        return {
            "_meta": {
                "description": "BRECQ 关键层重建 loss / grad；可直接编辑后运行 scripts/plot_recon_metrics.py 重绘",
                "created": datetime.now().isoformat(timespec="seconds"),
                "layers_file": self.layers_file,
                "whitelist": sorted(self.whitelist),
            },
            "stages": {},
        }

    def should_track(self, layer_name: str) -> bool:
        return layer_in_whitelist(layer_name, self.whitelist)

    def create_logger(self, layer_name: str) -> Optional["ReconMetricsLogger"]:
        if not self.should_track(layer_name):
            return None
        return ReconMetricsLogger(
            registry=self,
            layer_name=normalize_layer_name(layer_name),
            log_every=self.log_every,
            max_iters=self.max_iters,
        )

    def save_layer(self, layer_name: str, records: List[Dict]) -> None:
        if not records:
            return
        layer_name = normalize_layer_name(layer_name)
        stage_data = self._blob.setdefault("stages", {}).setdefault(self.stage, {})
        stage_data[layer_name] = {
            "series": records,
            "n_points": len(records),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self._blob["_meta"]["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.master_path, "w", encoding="utf-8") as f:
            json.dump(self._blob, f, indent=2, ensure_ascii=False)

        png_path = os.path.join(self.figures_dir, f"{_safe_filename(layer_name)}_curves.png")
        _plot_layer_curves(layer_name, records, png_path)
        print(
            f"[recon_metrics] {self.stage}/{layer_name}: "
            f"{len(records)} pts -> {self.master_path} + {png_path}"
        )


class ReconMetricsLogger:
    """单层迭代记录，结束时写入 Registry。"""

    def __init__(
        self,
        registry: ReconMetricsRegistry,
        layer_name: str,
        max_iters: Optional[int] = None,
        log_every: int = 1,
    ):
        self.registry = registry
        self.layer_name = layer_name
        self.max_iters = max_iters
        self.log_every = max(1, int(log_every))
        self.records: List[Dict] = []

    def should_log(self, iter_idx: int) -> bool:
        one_based = iter_idx + 1
        if self.max_iters is not None and one_based > self.max_iters:
            return False
        return one_based % self.log_every == 0

    def log(
        self,
        iter_idx: int,
        total: float,
        rec: float,
        round_loss: float,
        beta: float,
        grad_norm: float,
        lr: Optional[float] = None,
    ) -> None:
        if not self.should_log(iter_idx):
            return
        row = {
            "iter": int(iter_idx + 1),
            "total": float(total),
            "rec": float(rec),
            "round": float(round_loss),
            "beta": float(beta),
            "grad_norm": float(grad_norm),
        }
        if lr is not None:
            row["lr"] = float(lr)
        self.records.append(row)

    def flush(self) -> None:
        self.registry.save_layer(self.layer_name, self.records)


def parse_recon_plot_stages(stages_raw: str) -> Set[str]:
    stages_raw = (stages_raw or "act").strip().lower()
    if stages_raw in ("all", "both"):
        return {"weight", "act"}
    return {s.strip() for s in stages_raw.split(",") if s.strip()}


def recon_plot_stage_enabled(stage: str, enabled: Set[str]) -> bool:
    if stage in enabled:
        return True
    if "act" in enabled and (stage == "act" or stage.startswith("act_")):
        return True
    return False


class ReconPlotContext:
    """与 sample_diffusion_ddim / two_stage 共用的 BRECQ 曲线记录上下文。"""

    def __init__(self, args: Any):
        self.log_dir = (getattr(args, "recon_plot_dir", "") or "").strip()
        self.log_every = int(getattr(args, "recon_plot_every", 10))
        self.max_iters = int(getattr(args, "recon_plot_max_iters", 2000))
        layers_file = (
            getattr(args, "recon_plot_layers_file", "") or DEFAULT_LAYERS_FILE
        )
        self.layers_file = layers_file
        self.layer_list = load_layer_whitelist(
            getattr(args, "recon_plot_layers", "") or "",
            layers_file,
        )
        self.enabled_stages = parse_recon_plot_stages(
            getattr(args, "recon_plot_stages", "") or "act"
        )
        self._registries: Dict[str, ReconMetricsRegistry] = {}

    def active(self) -> bool:
        return bool(self.log_dir)

    def enabled(self, stage: str) -> bool:
        return self.active() and recon_plot_stage_enabled(stage, self.enabled_stages)

    def get_registry(self, stage: str) -> Optional[ReconMetricsRegistry]:
        if not self.enabled(stage):
            return None
        if stage not in self._registries:
            self._registries[stage] = ReconMetricsRegistry(
                log_dir=self.log_dir,
                stage=stage,
                layer_whitelist=self.layer_list,
                log_every=self.log_every,
                max_iters=self.max_iters,
                layers_file=self.layers_file,
            )
            import logging

            logging.getLogger(__name__).info(
                "BRECQ 曲线 [%s]：记录 %d 个关键层 -> %s",
                stage,
                len(self.layer_list),
                os.path.join(self.log_dir, MASTER_LOG_NAME),
            )
        return self._registries[stage]


def run_recon_model(
    qnn: Any,
    root_module: Any,
    kwargs: dict,
    plot_ctx: Optional[ReconPlotContext],
    stage_name: str,
    layer_recon_fn: Callable,
    block_recon_fn: Callable,
) -> None:
    """递归 block/layer 重建；仅在 plot_ctx 启用对应 stage 时记录指标。"""
    from qdiff.quant_block import BaseQuantBlock
    from qdiff.quant_layer import QuantModule

    registry = plot_ctx.get_registry(stage_name) if plot_ctx else None

    def _walk(model, prefix: str = "") -> None:
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            metrics_logger = registry.create_logger(full_name) if registry else None
            call_kwargs = dict(kwargs, metrics_logger=metrics_logger, recon_tag=full_name)
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction:
                    continue
                layer_recon_fn(qnn, module, **call_kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction:
                    continue
                block_recon_fn(qnn, module, **call_kwargs)
            else:
                _walk(module, full_name)

    _walk(root_module)


def replot_from_log(
    log_path: str,
    out_dir: str = "",
    stages: Optional[List[str]] = None,
    layers: Optional[List[str]] = None,
) -> None:
    """从 recon_metrics_log.json 重绘（可手改 JSON 后再跑）。"""
    log_path = os.path.abspath(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    base_out = out_dir or os.path.join(os.path.dirname(log_path), "figures_replot")
    layer_filter = {normalize_layer_name(x) for x in layers} if layers else None

    for stage, stage_data in blob.get("stages", {}).items():
        if stages and stage not in stages:
            continue
        fig_dir = os.path.join(base_out, stage)
        os.makedirs(fig_dir, exist_ok=True)
        for layer_name, entry in stage_data.items():
            if layer_filter and normalize_layer_name(layer_name) not in layer_filter:
                continue
            series = entry.get("series", entry) if isinstance(entry, dict) else entry
            if not series:
                continue
            png_path = os.path.join(fig_dir, f"{_safe_filename(layer_name)}_curves.png")
            _plot_layer_curves(layer_name, series, png_path)
            print(f"replotted {stage}/{layer_name} -> {png_path}")
