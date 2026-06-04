#!/usr/bin/env python3
"""Per-method panels: full k curve vs Ridge vs best-k-per-draw, one panel each.

Reads pls_components_tradeoff.csv (per-k PLS Spearman + ridge_baseline) and
best_k_vs_fixed_k.csv (the best-k-per-draw level), and draws one subplot per
model so you can see, for every k, how PLS compares to Ridge and to the
inflated Fig-1A best-k line.

Usage:
    python per_method_panels.py \
        --tradeoff-csv .../pls_components_tradeoff.csv \
        --bestk-csv    .../best_k_vs_fixed_k.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tradeoff-csv", type=Path, required=True)
    ap.add_argument("--bestk-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.tradeoff_csv.parent / "per_method_panels.png"

    rows = list(csv.DictReader(open(args.tradeoff_csv)))
    curves = defaultdict(list)   # model -> [(k, mean, sd)]
    ridge = {}
    for r in rows:
        curves[r["model"]].append((int(r["k"]), float(r["spearman_mean"]),
                                   float(r["spearman_std"])))
        if r.get("ridge_baseline") not in ("", "None", None):
            ridge[r["model"]] = float(r["ridge_baseline"])
    models = sorted(curves)

    bestk = {}
    if args.bestk_csv and args.bestk_csv.exists():
        for r in csv.DictReader(open(args.bestk_csv)):
            bestk[r["model"]] = float(r["best_k_per_draw"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        pts = sorted(curves[m])
        ks = [p[0] for p in pts]
        mu = np.array([p[1] for p in pts])
        sd = np.array([p[2] for p in pts])
        ax.plot(ks, mu, "-o", color="#3b6ea8", label="PLS (per fixed k)")
        ax.fill_between(ks, mu - sd, mu + sd, color="#3b6ea8", alpha=0.15)
        # mark the peak
        bi = int(np.argmax(mu))
        ax.plot(ks[bi], mu[bi], "o", color="#1a3e6e", ms=9)
        ax.annotate(f"peak k={ks[bi]}\n{mu[bi]:.3f}", (ks[bi], mu[bi]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
        if m in ridge:
            ax.axhline(ridge[m], color="#c0504d", ls="--", lw=1.5,
                       label=f"Ridge {ridge[m]:.3f}")
        if m in bestk:
            ax.axhline(bestk[m], color="#f0a202", ls=":", lw=1.5,
                       label=f"best-k/draw {bestk[m]:.3f}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("PLS components k (log2)")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Year Spearman (balanced, mean ± SD)")
    fig.suptitle("Per-method: PLS at every k vs Ridge (dashed) vs Fig-1A best-k (dotted)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
