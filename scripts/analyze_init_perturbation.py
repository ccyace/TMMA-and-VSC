"""Aggregate controlled initialization-perturbation runs.

Example:
  python scripts/analyze_init_perturbation.py --run_dirs runs/rho0 runs/rho025 ...
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def ols_quadratic(rho, y):
    x = np.column_stack([np.ones_like(rho), rho, rho ** 2])
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float(np.dot(residual, residual) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_beta2 = beta[2] / max(se[2], 1e-12)
    p_beta2 = float(2.0 * stats.t.sf(abs(t_beta2), dof))
    return beta, se, p_beta2


def early_gradient_variance(run_dir):
    """Mean per-layer variance of recorded early Adam gradient norms."""
    path = os.path.join(run_dir, "recon_metrics_log.json")
    if not os.path.isfile(path):
        return float("nan")
    blob = read_json(path)
    variances = []
    for stage, layers in blob.get("stages", {}).items():
        if not stage.startswith("act_"):
            continue
        for entry in layers.values():
            values = [x["grad_norm"] for x in entry.get("series", [])]
            if len(values) > 1:
                variances.append(float(np.var(values, ddof=1)))
    return float(np.mean(variances)) if variances else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="per-rho output directories of two_stage_quantized_sampling.py")
    p.add_argument("--out_dir", default="init_perturbation_analysis")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, all_rho, all_y, final_errors, direction_norms = [], [], [], {}, {}
    traj_curves = []
    for run_dir in args.run_dirs:
        s1 = read_json(os.path.join(run_dir, "init_perturbation_stage1_group0.json"))
        s2 = read_json(os.path.join(run_dir, "init_perturbation_stage2_group1.json"))
        tr = read_json(os.path.join(run_dir, "init_perturbation_trajectory.json"))
        rho = float(s1["init_perturb_rho"])
        sample_path = os.path.join(run_dir, tr["sample_file"])
        samples = np.load(sample_path)
        y = np.asarray(samples["final_mse"], dtype=np.float64)
        final_errors[rho] = np.asarray(samples["final_error"], dtype=np.float64)
        direction_sq = 0.0
        for stage_name in ("stage1_group0", "stage2_group1"):
            scale_blob = torch_load_cpu(os.path.join(run_dir, "init_scales_" + stage_name + ".pt"))
            for name, mm in scale_blob["minmax"].items():
                if name in scale_blob["tmma"]:
                    direction_sq += float((mm["delta"] - scale_blob["tmma"][name]["delta"]).pow(2).sum())
        direction_norms[rho] = direction_sq ** 0.5
        a1, a2 = s1["activation_probe"], s2["activation_probe"]
        rows.append({
            "rho": rho,
            "init_rel_l2_g0": s1["initialization"]["relative_l2_to_tmma"],
            "init_rel_l2_g1": s2["initialization"]["relative_l2_to_tmma"],
            "activation_mse_g0": a1["element_weighted_mse"],
            "activation_mse_g1": a2["element_weighted_mse"],
            "clip_rate_g0": a1["element_weighted_clip_rate"],
            "clip_rate_g1": a2["element_weighted_clip_rate"],
            "early_adam_grad_norm_var": early_gradient_variance(run_dir),
            "final_latent_mse": float(y.mean()),
            "final_latent_mse_se": float(y.std(ddof=1) / np.sqrt(max(len(y), 1))),
        })
        all_rho.extend([rho] * len(y))
        all_y.extend(y.tolist())
        traj_curves.append((rho, tr["steps"]))

    rows.sort(key=lambda x: x["rho"])
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rho_vec, y_vec = np.asarray(all_rho), np.asarray(all_y)
    beta, se, p_beta2 = ols_quadratic(rho_vec, y_vec)
    result = {
        "model": "E_final = beta0 + beta1 rho + beta2 rho^2",
        "beta0": float(beta[0]), "beta1": float(beta[1]), "beta2": float(beta[2]),
        "se_beta2": float(se[2]), "p_beta2": p_beta2,
        "interpretation": (
            "positive statistically supported curvature" if beta[2] > 0 and p_beta2 < 0.05
            else "no statistically supported positive quadratic curvature"
        ),
    }

    if 0.0 in final_errors:
        base = final_errors[0.0]
        gains = {}
        for rho, err in final_errors.items():
            if rho <= 0 or err.shape != base.shape:
                continue
            # Directional finite-difference gain in output-error space.  The
            # common initial scale direction is fixed by the experimental design.
            gains[str(rho)] = float(np.linalg.norm(err - base) / max(rho * direction_norms[rho], 1e-12))
        result["directional_gain_per_rho"] = gains

    with open(os.path.join(args.out_dir, "quadratic_fit.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    r = np.array([row["rho"] for row in rows])
    axes[0].plot(r, [row["activation_mse_g0"] for row in rows], "o-", label="group 0")
    axes[0].plot(r, [row["activation_mse_g1"] for row in rows], "s-", label="group 1")
    axes[0].set(xlabel="rho", ylabel="activation quantization MSE")
    axes[0].legend(); axes[0].grid(alpha=.3)
    ybar = np.array([row["final_latent_mse"] for row in rows])
    axes[1].errorbar(r, ybar, yerr=[row["final_latent_mse_se"] for row in rows], fmt="o", capsize=3)
    grid = np.linspace(r.min(), r.max(), 200)
    axes[1].plot(grid, beta[0] + beta[1] * grid + beta[2] * grid**2, label="quadratic fit")
    axes[1].set(xlabel="rho", ylabel="final latent MSE")
    axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(r, [row["early_adam_grad_norm_var"] for row in rows], "o-")
    axes[2].set(xlabel="rho", ylabel="early Adam grad-norm variance")
    axes[2].grid(alpha=.3)
    for rho, steps in sorted(traj_curves):
        axes[3].plot([x["t"] for x in steps], [x["latent_mse"] for x in steps], label="rho={}".format(rho))
    axes[3].invert_xaxis()
    axes[3].set(xlabel="DDIM timestep", ylabel="latent MSE")
    axes[3].legend(fontsize=7); axes[3].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mechanism_summary.png"), dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
