r"""
分布层对比实验：同一 FP 去噪网络，在「该池 + 该子采样规则」下估计参考边缘分布，
再度量各校准集设计相对参考的距离。

参考（金标准近似）：与 baseline 相同 allst 池，在 (k,i) 上均匀随机抽 ref_pairs 条 (x,t)，
每层每次前向再随机采 scalars_per_forward 个标量（与两侧设计同一规则），
得到大样本经验分布 ≈ 该池上均匀边缘分布的强近似。

用法示例（PowerShell）：
  python get_calibrations/compare_calib_activation_distribution.py `
    --config configs/cifar10.yml `
    --baseline_pt ...\cifar_sd1236_sample2048_allst.pt `
    --method_pt ...\cifar_dynamic_g0_half0.pt `
    --out_dir out/calib_dist_compare `
    --ref_pairs 50000 `
    --max_pairs_baseline 500 `
    --max_pairs_method 256 `
    --scalars_per_forward 4096

说明：
  - baseline：池中随机 max_pairs_baseline 个 (k,i)（随机校准对照）。
  - method：扁平校准集抽 max_pairs_method 条 (x,t)（规则同原 sample_method_pairs）。
  - 默认层：仅用 method_pt 的 meta.sensitive_layers（与动态脚本一致）；若无则须指定 --layers。
  - 输出：相对参考的 W1、KS、离散 KL（共享分箱 + Laplace）；baseline↔method 同上；平滑密度折线图 + summary.json。
  - --ref_pairs 0 可跳过参考（仅 baseline↔method，用于快速试跑）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import stats
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model


def dict2namespace(config):
    import argparse as ap

    ns = ap.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(ns, key, dict2namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def list_conv_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(stats.wasserstein_distance(a, b))
    except Exception:
        return float("nan")


def _shared_edges_from_samples(
    arrs: List[np.ndarray],
    n_bins: int,
    pct_lo: float = 0.5,
    pct_hi: float = 99.5,
) -> np.ndarray:
    """若干一维样本的公共直方图边界（按分位数裁剪长尾，避免 bin 浪费）。"""
    parts = [a.ravel() for a in arrs if a is not None and a.size > 0]
    if not parts:
        return np.linspace(-1.0, 1.0, n_bins + 1)
    allv = np.concatenate(parts)
    lo, hi = np.percentile(allv, [pct_lo, pct_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.min(allv)), float(np.max(allv))
    if lo >= hi:
        lo, hi = lo - 1e-3, hi + 1e-3
    return np.linspace(lo, hi, int(n_bins) + 1)


def _smoothed_hist_probs(samples: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    counts, _ = np.histogram(samples.astype(np.float64), bins=edges)
    p = counts.astype(np.float64) + float(alpha)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p, dtype=np.float64) / len(p)
        return p
    return p / s


def _kl_probs(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q)，p、q 已为正且归一化。"""
    p = np.clip(p.astype(np.float64), 1e-300, 1.0)
    q = np.clip(q.astype(np.float64), 1e-300, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _js_probs(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon 散度（自然底，上界 ln 2）。"""
    p = np.clip(p.astype(np.float64), 1e-300, 1.0)
    q = np.clip(q.astype(np.float64), 1e-300, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (_kl_probs(p, m) + _kl_probs(q, m))


def _kl_pair_on_shared_bins(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float]:
    """返回 KL(a||b), KL(b||a), JS(a,b)。"""
    pa = _smoothed_hist_probs(a, edges, alpha)
    pb = _smoothed_hist_probs(b, edges, alpha)
    return _kl_probs(pa, pb), _kl_probs(pb, pa), _js_probs(pa, pb)


def _plot_smoothed_density_lines(
    ax,
    series: List[Tuple[np.ndarray, str, str]],
    n_bins: int,
    smooth_sigma: float,
) -> None:
    """同一坐标轴上多条平滑密度折线（共享分箱 + 对密度做 1D 高斯平滑）。"""
    arrs = [t[0] for t in series if t[0].size > 0]
    if not arrs:
        return
    edges = _shared_edges_from_samples(arrs, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    for samples, label, color in series:
        if samples.size == 0:
            continue
        dens, _ = np.histogram(samples.astype(np.float64), bins=edges, density=True)
        line = gaussian_filter1d(dens.astype(np.float64), sigma=float(smooth_sigma))
        ax.plot(centers, line, label=label, color=color, linewidth=1.8, alpha=0.92)


def collect_layer_scalars(
    model: nn.Module,
    device: torch.device,
    xs: torch.Tensor,
    ts: torch.Tensor,
    layer_names: List[str],
    batch_size: int,
    scalars_per_forward: int,
    seed: int,
    progress_label: str = "",
) -> Dict[str, np.ndarray]:
    """xs [B,3,H,W], ts [B] 或 [B] float，逐 batch 前向，hook 指定层输出并子采样标量。"""
    buffers: Dict[str, List[np.ndarray]] = {n: [] for n in layer_names}
    handles = []

    def make_hook(name: str):
        def _hook(_m, _inp, out):
            if not isinstance(out, torch.Tensor):
                return
            flat = out.detach().float().reshape(-1)
            n_take = min(int(flat.numel()), scalars_per_forward)
            if n_take <= 0:
                return
            idx = torch.randperm(flat.numel(), device=flat.device)[:n_take]
            buffers[name].append(flat[idx].cpu().numpy())

        return _hook

    modules_by_name = dict(model.named_modules())
    for name in layer_names:
        if name not in modules_by_name:
            raise KeyError(f"层不存在: {name}")
        m = modules_by_name[name]
        handles.append(m.register_forward_hook(make_hook(name)))

    n = xs.size(0)
    num_batches = (n + batch_size - 1) // batch_size
    log_every = max(1, num_batches // 25) if progress_label else 0

    model.eval()
    with torch.no_grad():
        for bi, start in enumerate(range(0, n, batch_size)):
            if progress_label and log_every and (
                bi == 0 or (bi + 1) % log_every == 0 or start + batch_size >= n
            ):
                print(
                    f"[{progress_label}] 前向 batch {bi + 1}/{num_batches} "
                    f"({100.0 * (bi + 1) / num_batches:.0f}%)",
                    flush=True,
                )
            xb = xs[start : start + batch_size].to(device)
            tb = ts[start : start + batch_size].to(device)
            _ = model(xb, tb)

    for h in handles:
        h.remove()

    out_dict: Dict[str, np.ndarray] = {}
    for name in layer_names:
        if not buffers[name]:
            out_dict[name] = np.array([], dtype=np.float32)
        else:
            out_dict[name] = np.concatenate(buffers[name], axis=0).astype(np.float32)
    return out_dict


def sample_allst_pairs(
    xs: torch.Tensor,
    ts: torch.Tensor,
    num_pairs: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """xs [T,N,C,H,W], ts [T,N]，随机 (k,i)。"""
    rng = np.random.RandomState(seed)
    T, N = xs.shape[0], xs.shape[1]
    k = rng.randint(0, T, size=num_pairs)
    i = rng.randint(0, N, size=num_pairs)
    x_out = xs[k, i].float()
    t_out = ts[k, i].float().view(-1)
    return x_out, t_out


def sample_method_pairs(
    d: dict,
    num_pairs: int,
    seed: int,
    oversample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = d["xs"].float()
    ts = d["ts"].float().view(-1)
    B = xs.size(0)
    rng = np.random.RandomState(seed + 17)
    if B >= num_pairs:
        idx = rng.choice(B, size=num_pairs, replace=False)
    else:
        if oversample:
            idx = rng.choice(B, size=num_pairs, replace=True)
        else:
            idx = np.arange(B, dtype=np.int64)
    return xs[idx], ts[idx]


def default_layers_from_meta(method_pt: str) -> Optional[List[str]]:
    method_pt = method_pt.strip()
    d = torch.load(method_pt, map_location="cpu")
    meta = d.get("meta") if isinstance(d, dict) else None
    if isinstance(meta, dict) and meta.get("sensitive_layers"):
        return list(meta["sensitive_layers"])
    return None


def _abs_same(a: str, b: str) -> bool:
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cifar10.yml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline_pt", type=str, required=True, help="allst 格式 .pt（随机 baseline 的池）")
    p.add_argument(
        "--pool_pt",
        type=str,
        default="",
        help="可选：构建参考分布时用的 allst .pt；默认同 baseline_pt",
    )
    p.add_argument("--method_pt", type=str, required=True, help="扁平 xs/ts 校准 .pt（ours）")
    p.add_argument("--out_dir", type=str, default="out/calib_dist_compare")
    p.add_argument(
        "--ref_pairs",
        type=int,
        default=50000,
        help="参考边缘分布用的 (k,i) 条数（越大越接近池上均匀边缘）；0 表示不算参考",
    )
    p.add_argument(
        "--max_pairs_baseline",
        type=int,
        default=500,
        help="baseline random（allst 均匀抽）使用的 (x,t) 条数",
    )
    p.add_argument(
        "--max_pairs_method",
        type=int,
        default=500,
        help="method 扁平校准集抽样条数（与 oversample 逻辑配合）",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--scalars_per_forward", type=int, default=4096, help="每层每次前向最多采多少标量")
    p.add_argument(
        "--layers",
        type=str,
        default="",
        help="逗号分隔层名；空则仅用 method_pt 的 meta.sensitive_layers",
    )
    p.add_argument(
        "--conv_fallback_k",
        type=int,
        default=0,
        help="无 sensitive_layers 时，改用模型中前 k 个 Conv（默认 0=不 fallback，须显式传 --layers）",
    )
    p.add_argument(
        "--no_oversample_method",
        action="store_true",
        help="method 条数不足 max_pairs_method 时不重复抽样（可与 baseline 条数不等；若勾选会裁到 min 以对齐 baseline↔method）",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="不打印前向 batch 进度（默认会打印，ref_pairs 大时便于确认未卡死）",
    )
    p.add_argument(
        "--density_bins",
        type=int,
        default=200,
        help="密度折线图与离散 KL 共用的直方图 bin 数",
    )
    p.add_argument(
        "--density_smooth_sigma",
        type=float,
        default=2.5,
        help="对直方图密度曲线做 1D 高斯平滑的 sigma（以 bin 为单位）",
    )
    p.add_argument(
        "--kl_alpha",
        type=float,
        default=1.0,
        help="每个 bin 的 Laplace 伪计数，使离散 KL 有限且稳定",
    )
    args = p.parse_args()
    # 去掉首尾空白与误粘贴的换行，避免 Windows 下 open 报 OSError: [Errno 22]
    args.config = args.config.strip()
    args.baseline_pt = args.baseline_pt.strip()
    args.pool_pt = args.pool_pt.strip()
    args.method_pt = args.method_pt.strip()
    args.out_dir = args.out_dir.strip()
    args.device = args.device.strip()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = getattr(config, "split_shortcut", False)

    model = Model(config)
    ckpt = get_ckpt_path("ema_cifar10")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.to(device)
    model.eval()

    base = torch.load(args.baseline_pt, map_location="cpu")
    meth = torch.load(args.method_pt, map_location="cpu")
    xs_b, ts_b = base["xs"], base["ts"]
    if xs_b.dim() != 5 or ts_b.dim() != 2:
        raise ValueError("baseline 需为 allst：xs [T,N,C,H,W], ts [T,N]")
    xs_b = xs_b[: ts_b.shape[0]]

    pool_path = args.pool_pt or args.baseline_pt
    if _abs_same(pool_path, args.baseline_pt):
        xs_pool, ts_pool = xs_b, ts_b
    else:
        pool_d = torch.load(pool_path, map_location="cpu")
        xs_pool, ts_pool = pool_d["xs"], pool_d["ts"]
        if xs_pool.dim() != 5 or ts_pool.dim() != 2:
            raise ValueError("pool_pt 需为 allst：xs [T,N,C,H,W], ts [T,N]")
        xs_pool = xs_pool[: ts_pool.shape[0]]

    if args.layers.strip():
        layer_names = [x.strip() for x in args.layers.split(",") if x.strip()]
    else:
        layer_names = default_layers_from_meta(args.method_pt)
        if not layer_names:
            k = int(args.conv_fallback_k)
            if k <= 0:
                raise ValueError(
                    "method_pt 无 meta.sensitive_layers：请传 --layers，或设置 --conv_fallback_k>0"
                )
            layer_names = list_conv_names(model)[:k]

    for name in layer_names:
        if name not in dict(model.named_modules()):
            raise KeyError(f"层不存在: {name}")

    n_pb = int(args.max_pairs_baseline)
    n_pm = int(args.max_pairs_method)
    x_base, t_base = sample_allst_pairs(xs_b, ts_b, n_pb, args.seed)
    oversample = not args.no_oversample_method
    x_meth, t_meth = sample_method_pairs(meth, n_pm, args.seed, oversample=oversample)
    n_m = x_meth.size(0)
    n_b = x_base.size(0)
    if n_m != n_b and args.no_oversample_method:
        n_use = min(n_m, n_b)
        x_base, t_base = x_base[:n_use], t_base[:n_use]
        x_meth, t_meth = x_meth[:n_use], t_meth[:n_use]

    use_ref = int(args.ref_pairs) > 0
    x_ref: Optional[torch.Tensor] = None
    t_ref: Optional[torch.Tensor] = None
    if use_ref:
        x_ref, t_ref = sample_allst_pairs(
            xs_pool, ts_pool, int(args.ref_pairs), args.seed + 1000
        )

    summary: Dict[str, dict] = {
        "_meta": {
            "baseline_pt": os.path.abspath(args.baseline_pt),
            "pool_pt": os.path.abspath(pool_path),
            "method_pt": os.path.abspath(args.method_pt),
            "reference_rule": "uniform (k,i) on allst pool + same per-forward scalar subsampling",
            "ref_pairs": int(args.ref_pairs) if use_ref else 0,
            "n_baseline": int(x_base.size(0)),
            "n_method": int(x_meth.size(0)),
            "max_pairs_baseline": int(args.max_pairs_baseline),
            "max_pairs_method": int(args.max_pairs_method),
            "layers": layer_names,
            "scalars_per_forward": args.scalars_per_forward,
            "density_bins": args.density_bins,
            "density_smooth_sigma": args.density_smooth_sigma,
            "kl_alpha": args.kl_alpha,
            "kl_note": "discrete KL on shared histogram bins + Laplace(alpha); JS symmetric",
        }
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    seed_ref = args.seed + 50
    seed_base = args.seed + 51
    seed_meth = args.seed + 52

    def _prog(label: str) -> str:
        return "" if args.quiet else label

    if not args.quiet:
        n_ref = int(x_ref.size(0)) if use_ref and x_ref is not None else 0
        print(
            f"收集层激活子样本: ref_pairs={n_ref}, baseline={x_base.size(0)}, "
            f"method={x_meth.size(0)}, |layers|={len(layer_names)}, device={device}",
            flush=True,
        )

    s_ref_dict: Optional[Dict[str, np.ndarray]] = None
    if use_ref and x_ref is not None and t_ref is not None:
        s_ref_dict = collect_layer_scalars(
            model,
            device,
            x_ref,
            t_ref,
            layer_names,
            args.batch_size,
            args.scalars_per_forward,
            seed_ref,
            progress_label=_prog("ref"),
        )
    s_base_dict = collect_layer_scalars(
        model,
        device,
        x_base,
        t_base,
        layer_names,
        args.batch_size,
        args.scalars_per_forward,
        seed_base,
        progress_label=_prog("baseline"),
    )
    s_meth_dict = collect_layer_scalars(
        model,
        device,
        x_meth,
        t_meth,
        layer_names,
        args.batch_size,
        args.scalars_per_forward,
        seed_meth,
        progress_label=_prog("method"),
    )

    for layer in layer_names:
        s_base = s_base_dict[layer]
        s_meth = s_meth_dict[layer]
        if s_base.size == 0 or s_meth.size == 0:
            summary[layer] = {"error": "empty activations"}
            continue
        ks_bm = stats.ks_2samp(s_base, s_meth, alternative="two-sided", mode="auto")
        w1_bm = _w1(s_base, s_meth)

        edge_src: List[np.ndarray] = [s_base, s_meth]
        sr: Optional[np.ndarray] = None
        if use_ref and s_ref_dict is not None:
            t_sr = s_ref_dict[layer]
            if t_sr.size > 0:
                sr = t_sr
                edge_src.append(sr)

        edges = _shared_edges_from_samples(edge_src, args.density_bins)
        kl_b2m, kl_m2b, js_bm = _kl_pair_on_shared_bins(
            s_base, s_meth, edges, args.kl_alpha
        )

        entry: Dict = {
            "baseline_vs_method": {
                "ks_statistic": float(ks_bm.statistic),
                "ks_pvalue": float(ks_bm.pvalue),
                "wasserstein_1": w1_bm,
                "kl_baseline_to_method": kl_b2m,
                "kl_method_to_baseline": kl_m2b,
                "js_baseline_method": js_bm,
            },
            "n_scalars_baseline": int(s_base.size),
            "n_scalars_method": int(s_meth.size),
        }

        if use_ref and s_ref_dict is not None:
            s_r = s_ref_dict[layer]
            if s_r.size == 0:
                entry["reference"] = {"error": "empty reference activations"}
            else:
                ks_br = stats.ks_2samp(s_r, s_base, alternative="two-sided", mode="auto")
                ks_mr = stats.ks_2samp(s_r, s_meth, alternative="two-sided", mode="auto")
                kl_rb, kl_br, js_rb = _kl_pair_on_shared_bins(
                    s_r, s_base, edges, args.kl_alpha
                )
                kl_rm, kl_mr, js_rm = _kl_pair_on_shared_bins(
                    s_r, s_meth, edges, args.kl_alpha
                )
                entry["to_reference"] = {
                    "w1_baseline": _w1(s_r, s_base),
                    "w1_method": _w1(s_r, s_meth),
                    "ks_baseline": {
                        "statistic": float(ks_br.statistic),
                        "pvalue": float(ks_br.pvalue),
                    },
                    "ks_method": {
                        "statistic": float(ks_mr.statistic),
                        "pvalue": float(ks_mr.pvalue),
                    },
                    "kl_ref_to_baseline": kl_rb,
                    "kl_baseline_to_ref": kl_br,
                    "js_ref_baseline": js_rb,
                    "kl_ref_to_method": kl_rm,
                    "kl_method_to_ref": kl_mr,
                    "js_ref_method": js_rm,
                    "n_scalars_reference": int(s_r.size),
                }

        summary[layer] = entry

        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            series_plot: List[Tuple[np.ndarray, str, str]] = []
            series_plot.append((s_base, "ours", "#1f77b4"))
            series_plot.append((s_meth, "Random Sampling", "#ff7f0e"))
            if sr is not None and sr.size > 0 and x_ref is not None:
                series_plot.append((sr, "Reference", "#d62728"))
            _plot_smoothed_density_lines(
                ax,
                series_plot,
                n_bins=args.density_bins,
                smooth_sigma=args.density_smooth_sigma,
            )
            ax.set_title(layer, fontsize=16, pad=8)
            ax.set_xlabel("activation", fontsize=14)
            ax.set_ylabel("smoothed density", fontsize=14)
            ax.legend(loc="upper right", fontsize=14, framealpha=0.95)
            fig.tight_layout()
            safe = layer.replace(".", "_")
            fig.savefig(os.path.join(args.out_dir, f"{safe}_hist.png"), dpi=150)
            plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Done. Output:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
