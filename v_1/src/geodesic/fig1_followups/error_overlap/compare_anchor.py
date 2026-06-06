#!/usr/bin/env python3
"""Compare one ANCHOR model (default TF-IDF) against every other model, at the
fragment level and by metadata — to say "the anchor is strong on X, weak on Y".

Two kinds of output:

  1. venn_<anchor>_vs_<model>.png — a 2-set Venn of the CORRECT-fragment sets
     (both right / only-anchor / only-other / neither), with each disagreement
     region broken down by metadata (period composition shown as a stacked bar),
     so you can see *what kind* of fragment each side uniquely gets right.

  2. delta_<label>.png — heatmap of (anchor_acc - other_acc) per metadata group:
     blue = anchor stronger, red = anchor weaker, one column per rival model.
     The literal "anchor strong on these values, weak on those" table.

Usage:
    python compare_anchor.py --pred-csv .../predictions.csv --anchor tfidf --tol 100
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

META = ["period", "sub_genre", "ruler", "provenance"]
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
            "#56B4E9", "#999999", "#F0E442", "#882255", "#44AA99"]


def _setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })


def load(pred_csv, tol):
    rows = list(csv.DictReader(open(pred_csv)))
    models = [c[5:] for c in rows[0] if c.startswith("pred_")]
    for r in rows:
        yt = float(r["year_true"])
        for m in models:
            v = r[f"pred_{m}"]
            r[f"_ok_{m}"] = v not in ("", "nan") and abs(float(v) - yt) <= tol
    return rows, models


def venn_pair(rows, anchor, other, color_by, out, plt, tol):
    import matplotlib.patches as mp
    _setup(plt)
    A = {r["fragment_id"] for r in rows if r[f"_ok_{anchor}"]}
    B = {r["fragment_id"] for r in rows if r[f"_ok_{other}"]}
    by_id = {r["fragment_id"]: r for r in rows}
    both, onlyA, onlyB = A & B, A - B, B - A
    neither = {r["fragment_id"] for r in rows} - (A | B)

    fig, (axv, axb) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # --- Venn circles ---
    axv.add_patch(mp.Circle((-0.42, 0), 0.85, color=_PALETTE[3], alpha=0.30))
    axv.add_patch(mp.Circle((0.42, 0), 0.85, color=_PALETTE[0], alpha=0.30))
    axv.text(-0.95, 0, f"only\n{anchor}\n{len(onlyA)}", ha="center", va="center", fontsize=11)
    axv.text(0.95, 0, f"only\n{other}\n{len(onlyB)}", ha="center", va="center", fontsize=11)
    axv.text(0, 0, f"both\n{len(both)}", ha="center", va="center", fontsize=12, fontweight="bold")
    axv.text(0, -1.15, f"neither: {len(neither)}", ha="center", fontsize=10, color="#666")
    axv.text(-0.42, 0.95, anchor, ha="center", fontsize=11, fontweight="bold", color=_PALETTE[3])
    axv.text(0.42, 0.95, other, ha="center", fontsize=11, fontweight="bold", color=_PALETTE[0])
    axv.set_xlim(-1.8, 1.8); axv.set_ylim(-1.4, 1.4); axv.axis("off")
    axv.set_title(f"Correct-fragment sets (±{tol:.0f} yr)")

    # --- composition of the disagreement regions, colored by `color_by` ---
    regions = [(f"only {anchor}\n(n={len(onlyA)})", onlyA),
               (f"both\n(n={len(both)})", both),
               (f"only {other}\n(n={len(onlyB)})", onlyB),
               (f"neither\n(n={len(neither)})", neither)]
    cats = [c for c, _ in Counter(r[color_by] for r in rows).most_common()]
    cmap = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}
    x = np.arange(len(regions))
    bottoms = np.zeros(len(regions))
    for c in cats:
        vals = np.array([sum(by_id[i][color_by] == c for i in ids) / max(1, len(ids))
                         for _, ids in regions])
        axb.bar(x, vals, bottom=bottoms, color=cmap[c], label=str(c)[:22], edgecolor="white", lw=0.4)
        bottoms += vals
    axb.set_xticks(x); axb.set_xticklabels([r[0] for r in regions], fontsize=8)
    axb.set_ylabel(f"composition by {color_by}")
    axb.set_title(f"What's in each region ({color_by})")
    axb.grid(axis="x", visible=False); axb.set_ylim(0, 1)
    axb.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{anchor}  vs  {other}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"venn_{anchor}_vs_{other}.png", bbox_inches="tight"); plt.close(fig)
    print(f"[ok] venn_{anchor}_vs_{other}.png   "
          f"both={len(both)} only_{anchor}={len(onlyA)} only_{other}={len(onlyB)} neither={len(neither)}")


def delta_heatmap(rows, anchor, others, label, out, plt, tol, min_n=10, top_k=16):
    _setup(plt)
    vals = np.array([r.get(label, "") for r in rows])
    groups, counts = np.unique(vals, return_counts=True)
    keep = [g for g, c in sorted(zip(groups, counts), key=lambda x: -x[1])
            if c >= min_n and g != ""][:top_k]
    if not keep:
        return

    def acc(model, g):
        grp = [r for r in rows if r.get(label, "") == g]
        return np.mean([r[f"_ok_{model}"] for r in grp]) if grp else np.nan

    M = np.array([[acc(anchor, g) - acc(o, g) for o in others] for g in keep])
    fig, ax = plt.subplots(figsize=(1.7 * len(others) + 3.5, 0.5 * len(keep) + 2))
    lim = np.nanmax(np.abs(M)) or 0.1
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels([f"vs {o}" for o in others], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([f"{str(g)[:24]} (n={int((vals==g).sum())})" for g in keep], fontsize=8)
    for i in range(len(keep)):
        for j in range(len(others)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > lim * 0.6 else "black")
    ax.set_title(f"{anchor} accuracy − rival, by {label}\nblue = {anchor} stronger · red = weaker")
    fig.colorbar(im, label=f"Δ frac correct (±{tol:.0f} yr)")
    ax.grid(False); fig.tight_layout()
    fig.savefig(out / f"delta_{label}_{anchor}.png", bbox_inches="tight"); plt.close(fig)
    print(f"[ok] delta_{label}_{anchor}.png ({len(keep)} groups)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--anchor", default="tfidf")
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--color-by", default="period", help="metadata for Venn region composition")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=16)
    args = ap.parse_args()
    out = args.out_dir or args.pred_csv.parent
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, models = load(args.pred_csv, args.tol)
    if args.anchor not in models:
        raise SystemExit(f"anchor '{args.anchor}' not in {models}")
    others = [m for m in models if m != args.anchor]

    for o in others:
        venn_pair(rows, args.anchor, o, args.color_by, out, plt, args.tol)
    for label in [c for c in META if c in rows[0]]:
        delta_heatmap(rows, args.anchor, others, label, out, plt, args.tol,
                      min_n=args.min_n, top_k=args.top_k)


if __name__ == "__main__":
    main()
