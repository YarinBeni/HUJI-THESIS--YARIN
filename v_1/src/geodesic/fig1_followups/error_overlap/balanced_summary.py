#!/usr/bin/env python3
"""Scale-aware summary + chronology for any predictions_*/predictions.csv.

Regenerates (committed replacement for the prior laptop-only scripts) — works on
any predictions dir, auto-detecting models from pred_<m> columns:

  - summary.csv          Spearman, MAE, MAE/std, acc@25 + lift-vs-dummy, + dummy row
  - accuracy_at_N_table.csv  acc@N for N in {25,50,75,100} + Δ-vs-dummy + avg row
  - accuracy_at_N.png    acc@N line per model + dummy floor
  - chronology.png       (A) accuracy-vs-tolerance sweep + dummy; (B) per-ruler MAE
                         vs chronological order + per-ruler dummy-MAE line

Dummy = predict-the-constant (median year) — the MAE-optimal baseline. On the
balanced sets the year span is tiny (~188 yr) so acc@100 is near-trivial; always
read acc@25 vs the dummy floor, plus MAE/std and Spearman.

Usage:
    python balanced_summary.py --pred-csv .../predictions_tier0_balanced/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
NS = [25, 50, 75, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.pred_csv.parent
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    models = [c[5:] for c in df.columns if c.startswith("pred_")]
    cm = {m: PAL[i % len(PAL)] for i, m in enumerate(models)}
    y = df.year_true.values
    dummy = np.median(y)
    derr = np.abs(y - dummy)
    dmae, dstd = derr.mean(), np.abs(y - y.mean()).std()

    def stats(m):
        ok = df[f"pred_{m}"].notna()
        pr, yy = df[f"pred_{m}"][ok].values, y[ok]
        ae = np.abs(pr - yy)
        return dict(sp=spearmanr(pr, yy).statistic, mae=ae.mean(), mae_std=ae.std(),
                    acc={N: (ae <= N).mean() for N in NS})

    S = {m: stats(m) for m in models}

    # summary.csv
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "spearman", "mae", "mae_over_std", "acc@25", "acc@25_lift_vs_dummy"])
        d_acc25 = (derr <= 25).mean()
        for m in models:
            s = S[m]
            w.writerow([m, f"{s['sp']:.3f}", f"{s['mae']:.1f}", f"{s['mae']/dstd:.3f}",
                        f"{s['acc'][25]:.3f}", f"{s['acc'][25]-d_acc25:+.3f}"])
        w.writerow(["DUMMY(median)", "0.000", f"{dmae:.1f}", f"{dmae/dstd:.3f}", f"{d_acc25:.3f}", "+0.000"])
    print(f"[ok] summary.csv  (dummy: MAE {dmae:.1f}, acc@25 {d_acc25:.3f})")

    # accuracy_at_N_table.csv
    with open(out / "accuracy_at_N_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"acc@{N}" for N in NS] + [f"Δ@{N}" for N in NS] + ["avg_lift"])
        for m in models:
            d = [(derr <= N).mean() for N in NS]
            a = [S[m]["acc"][N] for N in NS]
            lifts = [a[i] - d[i] for i in range(len(NS))]
            w.writerow([m] + [f"{v:.3f}" for v in a] + [f"{v:+.3f}" for v in lifts]
                       + [f"{np.mean(lifts):+.3f}"])
        w.writerow(["DUMMY(median)"] + [f"{(derr<=N).mean():.3f}" for N in NS]
                   + ["+0.000"] * len(NS) + ["+0.000"])
    print("[ok] accuracy_at_N_table.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "axes.axisbelow": True,
                         "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "legend.frameon": False})

    # accuracy_at_N.png
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        ax.plot(NS, [S[m]["acc"][N] for N in NS], "-o", lw=2, color=cm[m], label=m)
    ax.plot(NS, [(derr <= N).mean() for N in NS], "--s", lw=2, color="#888888", label="dummy (median)")
    ax.set_xlabel("tolerance ± years"); ax.set_ylabel("fraction within tolerance")
    ax.set_title("Accuracy @ N vs predict-the-mean floor")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "accuracy_at_N.png", bbox_inches="tight"); plt.close(fig)
    print("[ok] accuracy_at_N.png")

    # chronology.png — (A) accuracy vs continuous tolerance; (B) per-ruler MAE vs chronology
    tols = np.arange(5, 151, 5)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))
    for m in models:
        ok = df[f"pred_{m}"].notna(); ae = np.abs(df[f"pred_{m}"][ok].values - y[ok])
        axA.plot(tols, [(ae <= t).mean() for t in tols], lw=2, color=cm[m], label=m)
    axA.plot(tols, [(derr <= t).mean() for t in tols], "--", lw=2, color="#888888", label="dummy")
    axA.axvline(25, color="#bbbbbb", lw=1); axA.set_xlabel("tolerance ± years")
    axA.set_ylabel("fraction within tolerance"); axA.set_title("Accuracy vs tolerance (+ dummy floor)")
    axA.legend(fontsize=9)

    if "ruler" in df.columns:
        rulers = (df.groupby("ruler").year_true.median().sort_values(ascending=False))  # old->recent
        order = list(rulers.index)
        xpos = np.arange(len(order))
        for m in models:
            mae = [np.abs(df[(df.ruler == r)][f"pred_{m}"] - df[(df.ruler == r)].year_true).mean()
                   for r in order]
            axB.plot(xpos, mae, "-o", ms=5, lw=1.8, color=cm[m], label=m)
        dmae_r = [np.abs(df[df.ruler == r].year_true - dummy).mean() for r in order]
        axB.plot(xpos, dmae_r, "--", lw=2, color="#888888", label="dummy")
        axB.set_xticks(xpos)
        axB.set_xticklabels([f"{r[:12]}\n{int(rulers[r])}" for r in order], rotation=45, ha="right", fontsize=7)
        axB.set_ylabel("MAE (years)"); axB.set_title("Per-ruler MAE, oldest → most recent")
        axB.legend(fontsize=8)
    fig.suptitle(f"{out.name}: scale-aware dating quality", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out / "chronology.png", bbox_inches="tight"); plt.close(fig)
    print("[ok] chronology.png")


if __name__ == "__main__":
    main()
