#!/usr/bin/env python3
"""Presentation figure — WHERE dating fails, by period / object-type / year.

Reads predictions.csv (per-fragment OOF predictions, 4 models) and shows how the
"how many of the 4 models date this fragment within ±tol yr" composition breaks
down across metadata. The story: the universally-hard fragments are sparse early
periods + short-text objects, i.e. a corpus problem, not a model problem.

Panels:
  A. per-period stacked bar (fraction of fragments dated by 0..4 models)
  B. per-object-type (sub_genre) stacked bar, top groups by count
  C. fraction-correct (mean over models) vs year, binned by century

Usage:
    python plot_failure_breakdown.py \
        --pred-csv .../predictions/predictions.csv --tol 100
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NCORR_COLORS = ["#8B0000", "#E8743B", "#F0C808", "#9ACD32", "#1A7A1A"]  # 0..4 red->green


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=10, help="min fragments to show a group")
    ap.add_argument("--top-k", type=int, default=14)
    args = ap.parse_args()
    out = args.out or args.pred_csv.parent / "failure_breakdown.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = list(csv.DictReader(open(args.pred_csv)))
    models = [c[5:] for c in rows[0] if c.startswith("pred_")]

    def err(r, m):
        v = r[f"pred_{m}"]
        return abs(float(v) - float(r["year_true"])) if v not in ("", "nan") else np.nan

    for r in rows:
        r["_nc"] = int(sum(err(r, m) <= args.tol for m in models))
        r["_yr"] = float(r["year_true"])

    def stacked(ax, label):
        counts = Counter(r[label] for r in rows)
        keep = [g for g, _ in sorted(counts.items(), key=lambda x: -x[1])
                if counts[g] >= args.min_n][:args.top_k]
        comp = {g: np.zeros(len(models) + 1) for g in keep}
        for r in rows:
            if r[label] in comp:
                comp[r[label]][r["_nc"]] += 1
        keep.sort(key=lambda g: comp[g][len(models):].sum() / counts[g])  # hardest on top
        y = np.arange(len(keep))
        left = np.zeros(len(keep))
        for nc in range(len(models) + 1):
            vals = np.array([comp[g][nc] / counts[g] for g in keep])
            ax.barh(y, vals, left=left, color=NCORR_COLORS[nc])
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([f"{g[:26]} (n={counts[g]})" for g in keep], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of fragments")
        ax.set_title(f"by {label}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    stacked(axes[0], "period")
    stacked(axes[1], "sub_genre")

    # Panel C: fraction-correct vs century
    by_cent = defaultdict(list)
    for r in rows:
        by_cent[int(r["_yr"] // 100) * 100].append(r["_nc"] / len(models))
    cents = sorted(by_cent)
    means = [np.mean(by_cent[c]) for c in cents]
    ns = [len(by_cent[c]) for c in cents]
    axes[2].bar([str(c) for c in cents], means, color="#3b6ea8")
    for x, (m, n) in enumerate(zip(means, ns)):
        axes[2].text(x, m, f"n={n}", ha="center", va="bottom", fontsize=7)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel(f"mean fraction of models correct (±{args.tol:.0f} yr)")
    axes[2].set_xlabel("century (BCE, century start)")
    axes[2].set_title("dating accuracy vs age")
    axes[2].tick_params(axis="x", rotation=45)

    handles = [Patch(color=NCORR_COLORS[i], label=f"{i}/{len(models)} models")
               for i in range(len(models) + 1)]
    fig.legend(handles=handles, loc="upper center", ncol=len(models) + 1,
               fontsize=8, frameon=False, bbox_to_anchor=(0.36, 1.02))
    fig.suptitle("Where dating fails: sparse early periods + short-text objects "
                 "(shared across all 4 models)", y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
