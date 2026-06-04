#!/usr/bin/env python3
"""Apples-to-apples PLS-vs-Ridge — the fair version of Fig-1A.

Fig-1A lets PLS maximize over BOTH layer and k (k picked per-draw → optimistic),
while Ridge only maximizes over layer. That asymmetric selection is what makes
PLS look like it beats Ridge. This plot fixes PLS at a single k (matched to
Ridge's single knob) and compares, with SD whiskers, at each model's best layer.

Reads pls_components_tradeoff.csv (model,k,spearman_mean,spearman_std,
ridge_baseline) produced by aggregate_and_plot.py.

Usage:
    python fixed_k_pls_vs_ridge.py \
        --csv .../pls_ksweep/pls_components_tradeoff.csv --k 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3, help="fixed PLS n_components (default 3)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.csv.parent / f"fixed_k{args.k}_pls_vs_ridge.png"

    rows = list(csv.DictReader(open(args.csv)))
    models = sorted({r["model"] for r in rows})
    pls, pls_sd, ridge = {}, {}, {}
    for r in rows:
        if int(r["k"]) == args.k:
            pls[r["model"]] = float(r["spearman_mean"])
            pls_sd[r["model"]] = float(r["spearman_std"])
        if r["ridge_baseline"] not in ("", "None"):
            ridge[r["model"]] = float(r["ridge_baseline"])
    models = [m for m in models if m in pls]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 5))
    ax.bar(x - w / 2, [pls[m] for m in models], w, yerr=[pls_sd[m] for m in models],
           capsize=4, color="#3b6ea8", label=f"PLS (fixed k={args.k})")
    ax.bar(x + w / 2, [ridge[m] for m in models], w, color="#c0504d",
           label="Ridge (all columns)")
    for i, m in enumerate(models):
        ax.text(i - w / 2, pls[m], f"{pls[m]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ridge[m], f"{ridge[m]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD)")
    ax.set_title(f"Fair PLS-vs-Ridge: PLS at fixed k={args.k}, each at its best layer\n"
                 "(removes the per-draw best-k cherry-pick of Fig-1A)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[ok] wrote {out}")
    for m in models:
        verdict = "PLS>Ridge" if pls[m] > ridge[m] else "Ridge>=PLS"
        print(f"  {m:22s} PLS@k{args.k}={pls[m]:.3f}  Ridge={ridge[m]:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
