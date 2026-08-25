"""Trajectory deviation E_x(t) for Min-Max vs TMMA (the "latter" metric).

From the same x_T, run DDIM with FP / Min-Max / TMMA separately, then
  E_x(t) = (1/d) || x_t^Q - x_t^FP ||_2^2
averaged over samples.

Example:
  C:\\Users\\ASUS\\miniconda3\\envs\\qdiff\\python.exe scripts/eval_traj_error_minmax_tmma.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddim.functions.denoising import generalized_steps
from scripts.eval_pmse_minmax_tmma import (
    build_qnn,
    dict2namespace,
    load_fp_model,
    resume_quant_ckpt,
)

logger = logging.getLogger("eval_traj")


def get_beta_schedule(config, device):
    betas = np.linspace(
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.num_diffusion_timesteps,
        dtype=np.float64,
    )
    return torch.from_numpy(betas).float().to(device)


def build_ddim_seq(num_timesteps: int, timesteps: int, skip_type: str):
    if skip_type == "uniform":
        seq = list(range(0, num_timesteps, num_timesteps // timesteps))
        if len(seq) > timesteps:
            seq = seq[:timesteps]
    elif skip_type == "quad":
        seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), timesteps) ** 2).tolist()
        seq = [int(s) for s in seq]
    else:
        raise ValueError(skip_type)
    return seq


def state_timestep_labels(seq):
    """Map xs[k] from generalized_steps to the diffusion timestep of that state."""
    # xs[0] is the starting noisy state at t = seq[-1]
    # after each reverse step i->j, the appended state lives at j
    labels = [int(seq[-1])]
    seq_next = [-1] + list(seq[:-1])
    for j in reversed(seq_next):
        labels.append(int(j))
    return labels  # length = len(seq) + 1


@torch.no_grad()
def run_trajectory(model, x_T, seq, betas, eta=0.0):
    model.eval()
    xs, _ = generalized_steps(x_T, seq, model, betas, eta=eta)
    # generalized_steps keeps xs[0] on GPU and later states on CPU
    xs_cpu = [x.detach().cpu() for x in xs]
    return torch.stack(xs_cpu, dim=0)  # [S,B,C,H,W]


def per_step_mse(traj_q, traj_fp):
    """E_x for each step: mean over batch and spatial dims."""
    # [S,B,C,H,W]
    err = (traj_q.float() - traj_fp.float()).pow(2)
    # mean over B,C,H,W -> [S]
    return err.mean(dim=(1, 2, 3, 4)).numpy()


def parse_args():
    p = argparse.ArgumentParser(description="Trajectory E_x(t): Min-Max vs TMMA")
    p.add_argument("--config", type=str, default=str(_REPO_ROOT / "configs" / "cifar10.yml"))
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
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", type=str, default="quad", choices=["quad", "uniform"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_a_sym", action="store_true")
    p.add_argument("--no_split", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_REPO_ROOT / "outputs" / "traj_error_minmax_tmma"),
    )
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
    seq = build_ddim_seq(config.diffusion.num_diffusion_timesteps, args.timesteps, args.skip_type)
    t_labels = state_timestep_labels(seq)
    logger.info("DDIM seq len=%d, first_t=%d, last_state_t=%d", len(seq), seq[-1], t_labels[-1])

    # shared initial noise
    torch.manual_seed(args.seed)
    x_all = torch.randn(
        args.n_samples,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
    )

    # ---- FP trajectories ----
    logger.info("Running FP DDIM")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_fp = dict2namespace(yaml.safe_load(f))
    fp_model = load_fp_model(cfg_fp, device)
    fp_chunks = []
    for i in range(0, args.n_samples, args.batch_size):
        xb = x_all[i : i + args.batch_size].to(device)
        fp_chunks.append(run_trajectory(fp_model, xb, seq, betas, eta=args.eta))
    traj_fp = torch.cat(fp_chunks, dim=1)  # [S,N,C,H,W]
    del fp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "protocol": {
            "metric": "E_x(t)=(1/d)||x_t^Q - x_t^FP||_2^2",
            "same_x_T": True,
            "independent_ddim_updates": True,
            "n_samples": args.n_samples,
            "timesteps": args.timesteps,
            "skip_type": args.skip_type,
            "eta": args.eta,
            "seed": args.seed,
            "a_sym": a_sym,
            "split": split,
            "minmax_ckpt": args.minmax_ckpt,
            "tmma_ckpt": args.tmma_ckpt,
        },
        "methods": {},
    }

    curves = {}
    dummy_x = x_all[:1]
    dummy_t = torch.tensor([float(seq[-1])])

    for name, ckpt in (("minmax", args.minmax_ckpt), ("tmma", args.tmma_ckpt)):
        logger.info("=" * 60)
        logger.info("Running %s DDIM", name)
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_i = dict2namespace(yaml.safe_load(f))
        qnn = build_qnn(cfg_i, device, a_sym=a_sym, split=split)
        resume_quant_ckpt(qnn, ckpt, device, dummy_x, dummy_t)

        q_chunks = []
        for i in range(0, args.n_samples, args.batch_size):
            xb = x_all[i : i + args.batch_size].to(device)
            q_chunks.append(run_trajectory(qnn, xb, seq, betas, eta=args.eta))
        traj_q = torch.cat(q_chunks, dim=1)

        # per-sample final MSE for SE, and mean curve
        # E_x[s] averaged over samples
        err = (traj_q.float() - traj_fp.float()).pow(2).mean(dim=(2, 3, 4))  # [S,N]
        mean_curve = err.mean(dim=1).numpy()
        se_curve = err.std(dim=1, unbiased=True).numpy() / np.sqrt(err.shape[1])
        curves[name] = {"mean": mean_curve, "se": se_curve}

        results["methods"][name] = {
            "ckpt": ckpt,
            "e_x_final": float(mean_curve[-1]),
            "e_x_final_se": float(se_curve[-1]),
            "e_x_mean_over_steps": float(mean_curve.mean()),
        }
        np.save(out_dir / f"e_x_curve_{name}.npy", mean_curve)
        np.save(out_dir / f"e_x_se_{name}.npy", se_curve)
        logger.info(
            "%s E_x(final)=%.6e  mean_over_steps=%.6e",
            name, mean_curve[-1], mean_curve.mean(),
        )

        del qnn, traj_q
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t_arr = np.asarray(t_labels, dtype=np.float64)
    np.save(out_dir / "t_labels.npy", t_arr)

    # CSV
    with open(out_dir / "e_x_vs_timestep.csv", "w", encoding="utf-8") as f:
        f.write("t,minmax_e_x,minmax_se,tmma_e_x,tmma_se\n")
        for i, t in enumerate(t_arr):
            f.write(
                f"{t},{curves['minmax']['mean'][i]},{curves['minmax']['se'][i]},"
                f"{curves['tmma']['mean'][i]},{curves['tmma']['se'][i]}\n"
            )

    # Plot linear E_x only
    t_plot = np.where(t_arr < 0, 0.0, t_arr)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    for name, color, marker, label in [
        ("minmax", "#1f77b4", "o", "Min-Max"),
        ("tmma", "#d62728", "s", "TMMA"),
    ]:
        m, se = curves[name]["mean"], curves[name]["se"]
        ax.plot(t_plot, m, "-" + marker, ms=3, color=color, label=label)
        ax.fill_between(t_plot, m - se, m + se, color=color, alpha=0.15)
    ax.invert_xaxis()
    ax.set_xlabel("timestep t")
    ax.set_ylabel("Latent MSE")
    ax.set_title("Latent Trajectory Deviation during DDIM Sampling")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "e_x_vs_timestep.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "e_x_vs_timestep.pdf", bbox_inches="tight")

    mm = results["methods"]["minmax"]["e_x_final"]
    tm = results["methods"]["tmma"]["e_x_final"]
    results["summary"] = {
        "minmax_e_x_final": mm,
        "tmma_e_x_final": tm,
        "relative_reduction_final": float((mm - tm) / max(mm, 1e-12)),
        "tmma_lower_final": bool(tm < mm),
    }
    with open(out_dir / "e_x_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {out_dir / 'e_x_vs_timestep.png'}")


if __name__ == "__main__":
    main()
