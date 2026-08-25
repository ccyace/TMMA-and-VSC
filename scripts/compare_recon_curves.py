"""对比 baseline vs TMMA 激活 BRECQ 曲线 JSON。"""
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def load_stage(path, stage):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["stages"].get(stage, {})


def stats(series, window_points=50):
    early = series[:window_points]
    rec = [p["rec"] for p in early]
    gn = [p["grad_norm"] for p in early]
    rec_a = np.array(rec)
    gn_a = np.array(gn)
    return {
        "rec_early_mean": float(rec_a.mean()),
        "rec_early_std": float(rec_a.std()),
        "rec_final": float(series[-1]["rec"]),
        "rec_min": float(min(p["rec"] for p in series)),
        "grad_early_mean": float(gn_a.mean()),
        "grad_early_std": float(gn_a.std()),
        "grad_cv": float(gn_a.std() / gn_a.mean()) if gn_a.mean() > 1e-12 else 0.0,
        "grad_max_early": float(gn_a.max()),
    }


def main():
    base_path = _REPO / "baseline_recon_curves_run/recon_curves/recon_metrics_log.json"
    tmma_path = _REPO / "tmma_recon_curves_run/recon_curves/recon_metrics_log.json"
    base = load_stage(base_path, "act")
    tmma_g0 = load_stage(tmma_path, "act_g0")
    tmma_g1 = load_stage(tmma_path, "act_g1")
    layers = sorted(base.keys())

    print("=== 前50个记录点(约 iter 10-500) 对比 ===")
    hdr = "{:<26} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7}"
    print(hdr.format("layer", "B_rec", "T0_rec", "T1_rec", "B_gCV", "T0_gCV", "T1_gCV", "B_gMx", "T0_gMx"))
    print("-" * 95)
    rows = []
    for L in layers:
        b = stats(base[L]["series"])
        t0 = stats(tmma_g0[L]["series"]) if L in tmma_g0 else None
        t1 = stats(tmma_g1[L]["series"]) if L in tmma_g1 else None
        rows.append((L, b, t0, t1))
        print(
            hdr.format(
                L.replace("model.", ""),
                f"{b['rec_early_mean']:.4f}",
                f"{t0['rec_early_mean']:.4f}" if t0 else "-",
                f"{t1['rec_early_mean']:.4f}" if t1 else "-",
                f"{b['grad_cv']:.2f}",
                f"{t0['grad_cv']:.2f}" if t0 else "-",
                f"{t1['grad_cv']:.2f}" if t1 else "-",
                f"{b['grad_max_early']:.2f}",
                f"{t0['grad_max_early']:.2f}" if t0 else "-",
            )
        )

    def agg(stage_dict):
        rec, gcv, gmx = [], [], []
        for L, _, _, _ in rows:
            if L not in stage_dict:
                continue
            s = stats(stage_dict[L]["series"])
            rec.append(s["rec_early_mean"])
            gcv.append(s["grad_cv"])
            gmx.append(s["grad_max_early"])
        return np.mean(rec), np.mean(gcv), np.mean(gmx)

    print("\n=== 10 层平均 ===")
    for name, st in [("Baseline act", base), ("TMMA act_g0", tmma_g0), ("TMMA act_g1", tmma_g1)]:
        r, g, m = agg(st)
        print(f"  {name}: rec_early={r:.5f}, grad_CV={g:.3f}, grad_max_early={m:.3f}")

    w0 = w1 = wg0 = wg1 = 0
    for L, b, t0, t1 in rows:
        if t0 and t0["rec_early_mean"] < b["rec_early_mean"]:
            w0 += 1
        if t1 and t1["rec_early_mean"] < b["rec_early_mean"]:
            w1 += 1
        if t0 and t0["grad_cv"] < b["grad_cv"]:
            wg0 += 1
        if t1 and t1["grad_cv"] < b["grad_cv"]:
            wg1 += 1
    n = len(rows)
    print(f"\n早期 rec 更低: TMMA g0 {w0}/{n}, g1 {w1}/{n}")
    print(f"早期 grad_CV 更低(更稳): TMMA g0 {wg0}/{n}, g1 {wg1}/{n}")


if __name__ == "__main__":
    main()
