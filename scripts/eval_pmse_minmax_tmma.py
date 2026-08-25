"""Same-input Prediction MSE (PMSE) for Min-Max vs TMMA quantized U-Nets.

PMSE matches paper eq. (13) averaged over a held-out full-precision set H:
  PMSE = mean_{ (x_t,t) in H }  (1/d) || eps_Q(x_t,t) - eps_FP(x_t,t) ||_2^2

Example:
  C:\\Users\\ASUS\\miniconda3\\envs\\qdiff\\python.exe scripts/eval_pmse_minmax_tmma.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model
from qdiff import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import convert_adaround

logger = logging.getLogger("eval_pmse")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_fp_model(config, device):
    # Pretrained EMA U-Net is full-precision; keep the non-split shortcut path.
    config.split_shortcut = False
    model = Model(config)
    ckpt = get_ckpt_path("ema_cifar10")
    logger.info("FP ckpt: %s", ckpt)
    model.load_state_dict(torch_load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_qnn(config, device, *, weight_bit=8, act_bit=8, a_sym=True, split=True, sm_abit=8):
    config.split_shortcut = split
    fp = Model(config)
    # Dummy weights; overwritten by resume loader after wrapping.
    fp.to(device)
    fp.eval()
    wq = {"n_bits": weight_bit, "channel_wise": True, "scale_method": "max"}
    aq = {
        "n_bits": act_bit,
        "symmetric": a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }
    qnn = QuantModel(model=fp, weight_quant_params=wq, act_quant_params=aq, sm_abit=sm_abit)
    qnn.to(device)
    qnn.eval()
    return qnn


def resume_quant_ckpt(qnn, ckpt_path, device, dummy_x, dummy_t):
    """Device-aware version of resume_cali_model for W8A8 inference."""
    logger.info("Loading quantized ckpt: %s", ckpt_path)
    ckpt = torch_load(ckpt_path, map_location="cpu")

    qnn.set_quant_state(True, False)
    _ = qnn(dummy_x.to(device), dummy_t.to(device))
    convert_adaround(qnn)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

    ckpt_w = {k: v for k, v in ckpt.items() if "act" not in k}
    qnn.load_state_dict(ckpt_w, strict=True)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data
            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    qnn.set_quant_state(True, True)
    _ = qnn(dummy_x.to(device), dummy_t.to(device))

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer):
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)

    ckpt = torch_load(ckpt_path, map_location="cpu")
    qnn.load_state_dict(ckpt, strict=True)
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data
            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data
        elif isinstance(m, UniformAffineQuantizer):
            if m.zero_point is not None:
                zero_data = m.zero_point.item()
                delattr(m, "zero_point")
                assert int(zero_data) == zero_data
                m.zero_point = int(zero_data)

    qnn.eval()
    return qnn


def build_heldout_from_traj(
    traj_path,
    *,
    sample_start=1024,
    n_samples=256,
    step_stride=5,
    samples_per_step=4,
    seed=1234,
):
    """Build H from a saved FP trajectory, using later sample indices as hold-out."""
    blob = torch_load(traj_path, map_location="cpu")
    xs = blob["xs"]  # [T+1, N, C, H, W]
    ts = blob["ts"]  # [T, N]
    n_steps, n_total = ts.shape
    assert sample_start + n_samples <= n_total, (
        f"need samples [{sample_start}:{sample_start + n_samples}) but N={n_total}"
    )

    step_ids = list(range(0, n_steps, step_stride))
    rng = np.random.RandomState(seed)
    x_list, t_list, meta = [], [], []
    for si in step_ids:
        # xs[si] with ts[si]: state at the i-th reverse step of the saved FP trajectory.
        chosen = rng.choice(n_samples, size=min(samples_per_step, n_samples), replace=False)
        for off in chosen:
            idx = sample_start + int(off)
            x = xs[si, idx]
            t = ts[si, idx]
            x_list.append(x)
            t_list.append(t if t.ndim == 0 else t.reshape(()))
            meta.append({"step_index": int(si), "sample_index": int(idx), "t": float(t)})

    x_out = torch.stack(x_list, dim=0).float()
    t_out = torch.stack([tt.reshape(()) for tt in t_list], dim=0).float()
    logger.info(
        "Held-out H: %d pairs from %s (sample_start=%d, n_samples=%d, stride=%d, per_step=%d)",
        len(meta), traj_path, sample_start, n_samples, step_stride, samples_per_step,
    )
    return x_out, t_out, meta


def build_heldout_forward_cifar(config, device, betas, *, n_pairs=200, seed=1234):
    """Fallback: forward-diffuse CIFAR-10 test images at random DDIM timesteps."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(root=str(_REPO_ROOT / "data"), train=False, download=True, transform=transform)
    rng = np.random.RandomState(seed)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 100-step uniform DDIM schedule over 1000 diffusion steps
    ddim_steps = 100
    seq = list(range(0, 1000, 1000 // ddim_steps))[:ddim_steps]

    x_list, t_list, meta = [], [], []
    for i in range(n_pairs):
        img_idx = int(rng.randint(0, len(ds)))
        x0, _ = ds[img_idx]
        t_idx = int(seq[int(rng.randint(0, len(seq)))])
        eps = torch.randn_like(x0)
        a = alphas_cumprod[t_idx]
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * eps
        x_list.append(xt)
        t_list.append(torch.tensor(float(t_idx)))
        meta.append({"img_index": img_idx, "t": t_idx})

    return torch.stack(x_list), torch.stack(t_list), meta


@torch.no_grad()
def compute_pmse(fp_model, q_model, xs, ts, device, batch_size=32, meta=None):
    """Return per-pair MSE and summary stats (optionally stratified by early/late t)."""
    fp_model.eval()
    q_model.eval()
    msés = []
    for i in range(0, xs.shape[0], batch_size):
        xb = xs[i : i + batch_size].to(device)
        tb = ts[i : i + batch_size].to(device)
        eps_fp = fp_model(xb, tb)
        eps_q = q_model(xb, tb)
        err = (eps_q - eps_fp).float()
        per = err.reshape(err.shape[0], -1).pow(2).mean(dim=1)
        msés.append(per.cpu())
    per_mse = torch.cat(msés, dim=0).numpy()
    out = {
        "pmse": float(per_mse.mean()),
        "pmse_std": float(per_mse.std(ddof=1)) if len(per_mse) > 1 else 0.0,
        "pmse_se": float(per_mse.std(ddof=1) / np.sqrt(len(per_mse))) if len(per_mse) > 1 else 0.0,
        "n": int(len(per_mse)),
        "per_pair_mse": per_mse.tolist(),
    }
    if meta is not None and len(meta) == len(per_mse):
        t_vals = np.asarray([m["t"] for m in meta], dtype=np.float64)
        mid = 0.5 * (t_vals.max() + t_vals.min())
        early = per_mse[t_vals >= mid]  # larger t: earlier reverse steps
        late = per_mse[t_vals < mid]
        out["pmse_early_t"] = float(early.mean()) if len(early) else None
        out["pmse_late_t"] = float(late.mean()) if len(late) else None
        out["n_early"] = int(len(early))
        out["n_late"] = int(len(late))
    return out


def get_beta_schedule(config, device):
    betas = np.linspace(
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.num_diffusion_timesteps,
        dtype=np.float64,
    )
    return torch.from_numpy(betas).float().to(device)


def parse_args():
    p = argparse.ArgumentParser(description="PMSE: Min-Max vs TMMA, same FP inputs")
    p.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "cifar10.yml"),
    )
    p.add_argument(
        "--minmax_ckpt",
        type=str,
        default=r"C:\Users\ASUS\Desktop\ODE-scale\cifar_w8a8_ckpt.pth",
    )
    p.add_argument(
        "--tmma_ckpt",
        type=str,
        default=str(
            _REPO_ROOT
            / "two_quant_output_w8a8(50000，FID3.6413)"
            / "samples"
            / "2025-11-27-11-39-41"
            / "ckpt.pth"
        ),
    )
    p.add_argument(
        "--traj_path",
        type=str,
        default=r"C:\Users\ASUS\Desktop\ODE-scale\cifar_sd1236_sample2048_allst.pt",
        help="FP trajectory dump used to build held-out H (later sample indices)",
    )
    p.add_argument("--sample_start", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=256, help="hold-out sample pool size")
    p.add_argument("--step_stride", type=int, default=2, help="subsample DDIM steps")
    p.add_argument("--samples_per_step", type=int, default=4, help="hold-out samples drawn per DDIM step")
    p.add_argument("--n_pairs", type=int, default=0, help="if >0, randomly take this many pairs from pool")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--no_a_sym", action="store_true")
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--no_split", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_REPO_ROOT / "outputs" / "pmse_minmax_tmma"),
    )
    p.add_argument("--use_forward_cifar", action="store_true", help="fallback held-out construction")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))

    a_sym = not args.no_a_sym
    split = not args.no_split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = get_beta_schedule(config, device)
    if args.use_forward_cifar or not Path(args.traj_path).is_file():
        logger.info("Building held-out H via CIFAR-10 forward diffusion")
        xs, ts, meta = build_heldout_forward_cifar(
            config, device, betas.cpu(), n_pairs=max(args.n_pairs, 200), seed=args.seed
        )
    else:
        xs, ts, meta = build_heldout_from_traj(
            args.traj_path,
            sample_start=args.sample_start,
            n_samples=args.n_samples,
            step_stride=args.step_stride,
            samples_per_step=args.samples_per_step,
            seed=args.seed,
        )
        if args.n_pairs > 0 and args.n_pairs < len(meta):
            rng = np.random.RandomState(args.seed)
            pick = rng.choice(len(meta), size=args.n_pairs, replace=False)
            pick = np.sort(pick)
            xs, ts = xs[pick], ts[pick]
            meta = [meta[i] for i in pick]

    dummy_x = xs[:1]
    dummy_t = ts[:1]

    logger.info("Loading FP model")
    fp_model = load_fp_model(config, device)

    results = {
        "protocol": {
            "metric": "PMSE = mean (1/d)||eps_Q(x_t,t)-eps_FP(x_t,t)||_2^2",
            "same_input": True,
            "n_pairs": len(meta),
            "sample_start": args.sample_start,
            "n_samples_pool": args.n_samples,
            "step_stride": args.step_stride,
            "a_sym": a_sym,
            "split": split,
            "minmax_ckpt": args.minmax_ckpt,
            "tmma_ckpt": args.tmma_ckpt,
            "traj_path": args.traj_path,
            "seed": args.seed,
        },
        "methods": {},
    }

    for name, ckpt in (("minmax", args.minmax_ckpt), ("tmma", args.tmma_ckpt)):
        logger.info("=" * 60)
        logger.info("Evaluating %s", name)
        # Fresh config namespace each time because QuantModel mutates split_shortcut.
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_i = dict2namespace(yaml.safe_load(f))
        qnn = build_qnn(cfg_i, device, a_sym=a_sym, split=split)
        # Load FP weights into the wrapped model before resume (resume overwrites quant params).
        # QuantModel wraps a newly constructed Model; copy FP weights into org modules.
        # resume_quant_ckpt loads the full quantized state_dict, so FP init is not required.
        resume_quant_ckpt(qnn, ckpt, device, dummy_x, dummy_t)
        stats = compute_pmse(
            fp_model, qnn, xs, ts, device, batch_size=args.batch_size, meta=meta
        )
        results["methods"][name] = {
            "ckpt": ckpt,
            "pmse": stats["pmse"],
            "pmse_std": stats["pmse_std"],
            "pmse_se": stats["pmse_se"],
            "n": stats["n"],
            "pmse_early_t": stats.get("pmse_early_t"),
            "pmse_late_t": stats.get("pmse_late_t"),
            "n_early": stats.get("n_early"),
            "n_late": stats.get("n_late"),
        }
        logger.info(
            "%s PMSE=%.6e  (std=%.3e, se=%.3e, n=%d, early=%.3e, late=%.3e)",
            name, stats["pmse"], stats["pmse_std"], stats["pmse_se"], stats["n"],
            stats.get("pmse_early_t") or float("nan"),
            stats.get("pmse_late_t") or float("nan"),
        )
        # free VRAM between methods
        del qnn
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # keep per-pair for optional plots
        np.save(out_dir / f"per_pair_mse_{name}.npy", np.asarray(stats["per_pair_mse"]))

    mm = results["methods"]["minmax"]["pmse"]
    tm = results["methods"]["tmma"]["pmse"]
    mm_arr = np.load(out_dir / "per_pair_mse_minmax.npy")
    tm_arr = np.load(out_dir / "per_pair_mse_tmma.npy")
    paired = tm_arr - mm_arr
    # two-sided paired t-test without scipy dependency
    n = len(paired)
    mean_d = float(paired.mean())
    se_d = float(paired.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_stat = mean_d / max(se_d, 1e-12)
    results["summary"] = {
        "minmax_pmse": mm,
        "tmma_pmse": tm,
        "relative_reduction": float((mm - tm) / max(mm, 1e-12)),
        "tmma_lower": bool(tm < mm),
        "paired_mean_diff_tmma_minus_minmax": mean_d,
        "paired_diff_se": se_d,
        "paired_t_stat": float(t_stat),
        "note": (
            "Same-input PMSE supports finite-budget initialization sensitivity / "
            "error transmission; it does not by itself prove nonlinear amplification."
        ),
    }
    with open(out_dir / "pmse_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "heldout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {out_dir / 'pmse_summary.json'}")


if __name__ == "__main__":
    main()
