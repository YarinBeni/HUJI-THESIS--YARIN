#!/usr/bin/env python3
"""Balanced tier0 vs maximal — the 2x2 corner that separates LENGTH from CLASS.

Both inputs are class-balanced (200 MC draws, 8 rulers); they differ only in text
length (tier0 = full, maximal = <=32 words). So the tier0->maximal change is a
pure length effect with the dominant-class crutch already removed.

Reads predictions_tier0_balanced/ and predictions_maximal_balanced/, prints the
metric table, and writes balanced_length_compare.png:
  A. Spearman per model, tier0 vs maximal (+ predict-the-mean dummy floor)
  B. length sensitivity = Spearman(tier0) - Spearman(maximal) per model

Usage:
    python balanced_length_compare.py --base v_1/src/geodesic/fig1_followups/error_overlap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ["tfidf", "thalesian_cunei400m", "qwen3_32b"]
PAL = {"tier0": "#0072B2", "maximal": "#D55E00"}


def metrics(p: Path):
    df = pd.read_csv(p)
    y = df.year_true.values
    out = {}
    for m in MODELS:
        ok = df[f"pred_{m}"].notna()
        pr, yy = df[f"pred_{m}"][ok].values, y[ok]
        out[m] = dict(sp=spearmanr(pr, yy).statistic, mae=np.mean(np.abs(pr - yy)),
                      a25=np.mean(np.abs(pr - yy) <= 25))
    dummy_a25 = np.mean(np.abs(y - np.median(y)) <= 25)
    return out, dummy_a25, (y.min(), y.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("v_1/src/geodesic/fig1_followups/error_overlap"))
    args = ap.parse_args()
    out_png = args.base / "predictions_tier0_balanced" / "balanced_length_compare.png"

    t0, da0, span = metrics(args.base / "predictions_tier0_balanced/predictions.csv")
    mx, dam, _ = metrics(args.base / "predictions_maximal_balanced/predictions.csv")

    print(f"balanced span {span[0]:.0f}-{span[1]:.0f} BCE; dummy acc@25 ~{da0:.3f}")
    print(f"{'model':22s} {'t0 Sp':>6s} {'mx Sp':>6s} {'Δlen':>6s} {'t0 MAE':>7s} {'mx MAE':>7s}")
    for m in MODELS:
        print(f"{m:22s} {t0[m]['sp']:6.3f} {mx[m]['sp']:6.3f} {t0[m]['sp']-mx[m]['sp']:+6.3f} "
              f"{t0[m]['mae']:7.1f} {mx[m]['mae']:7.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "axes.axisbelow": True,
                         "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "legend.frameon": False})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MODELS)); w = 0.38
    for k, (clean, d) in enumerate([("tier0", t0), ("maximal", mx)]):
        sp = [d[m]["sp"] for m in MODELS]
        axA.bar(x + (k - 0.5) * w, sp, w, color=PAL[clean], label=f"balanced {clean}")
        for i, s in enumerate(sp):
            axA.text(x[i] + (k - 0.5) * w, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels(MODELS, rotation=15, ha="right")
    axA.set_ylabel("year Spearman"); axA.set_ylim(0, 0.55)
    axA.set_title("Both class-balanced: tier0 vs maximal\n(rank flips — tfidf needs length)")
    axA.legend(); axA.grid(axis="x", visible=False)

    dlen = [t0[m]["sp"] - mx[m]["sp"] for m in MODELS]
    axB.bar(x, dlen, color=["#8B0000" if v > 0.1 else "#999999" for v in dlen])
    for i, v in enumerate(dlen):
        axB.text(i, v, f"+{v:.3f}", ha="center", va="bottom", fontsize=9)
    axB.set_xticks(x); axB.set_xticklabels(MODELS, rotation=15, ha="right")
    axB.set_ylabel("length sensitivity  (Sp tier0 − Sp maximal)")
    axB.set_title("TF-IDF is length; thalesian/qwen are length-robust")
    axB.grid(axis="x", visible=False)
    fig.suptitle("Separating length from class: with classes balanced, TF-IDF's lead is "
                 "purely text length", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight")
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
