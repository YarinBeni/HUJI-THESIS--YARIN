#!/usr/bin/env python3
"""tier0 vs maximal — is the model difference just text length?

`maximal` truncates every fragment to <=32 words (std 502 -> 9), so comparing
the two cleanings isolates the length effect. Plots overall accuracy and the
length-dependence (Q4-Q1 spread) per model for both cleanings.

Usage:
    python tier0_vs_maximal.py --base v_1/src/geodesic/fig1_followups/error_overlap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["thalesian_cunei400m", "qwen3_32b", "tfidf"]
PAL = {"tier0": "#0072B2", "maximal": "#D55E00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("v_1/src/geodesic/fig1_followups/error_overlap"))
    ap.add_argument("--corpus", type=Path,
                    default=Path("v_1/data/evaluation/corpora/orcc_corpus.parquet"))
    ap.add_argument("--tol", type=float, default=100.0)
    args = ap.parse_args()
    out = args.base / "predictions_maximal" / "tier0_vs_maximal.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "axes.axisbelow": True,
                         "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "legend.frameon": False})

    corp = pd.read_parquet(args.corpus)[["fragment_id", "word_count"]]

    def load(p):
        df = pd.read_csv(p).merge(corp, on="fragment_id", how="left")
        for m in MODELS:
            df[f"ok_{m}"] = (df[f"pred_{m}"] - df["year_true"]).abs() <= args.tol
        df["wcq"] = pd.qcut(df.word_count, 4, labels=False)
        return df

    data = {"tier0": load(args.base / "predictions/predictions.csv"),
            "maximal": load(args.base / "predictions_maximal/predictions.csv")}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MODELS)); w = 0.38
    for k, (clean, df) in enumerate(data.items()):
        accs = [df[f"ok_{m}"].mean() for m in MODELS]
        axA.bar(x + (k - 0.5) * w, accs, w, color=PAL[clean], label=clean)
        for i, a in enumerate(accs):
            axA.text(x[i] + (k - 0.5) * w, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels(MODELS, rotation=15, ha="right")
    axA.set_ylim(0.6, 0.95); axA.set_ylabel(f"overall accuracy (±{args.tol:.0f} yr)")
    axA.set_title("Overall accuracy: tier0 vs maximal\n(length-hungry models drop)")
    axA.legend(); axA.grid(axis="x", visible=False)

    for k, (clean, df) in enumerate(data.items()):
        spreads = [df[df.wcq == 3][f"ok_{m}"].mean() - df[df.wcq == 0][f"ok_{m}"].mean()
                   for m in MODELS]
        axB.bar(x + (k - 0.5) * w, spreads, w, color=PAL[clean], label=clean)
        for i, s in enumerate(spreads):
            axB.text(x[i] + (k - 0.5) * w, s, f"{s:+.2f}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(MODELS, rotation=15, ha="right")
    axB.set_ylabel("length-dependence  (long Q4 − short Q1 accuracy)")
    axB.set_title("Length-dependence halves under maximal\n(truncation removes the long-text edge)")
    axB.axhline(0, color="#888888", lw=0.8); axB.legend(); axB.grid(axis="x", visible=False)
    fig.suptitle("Is the model difference just length? Largely yes — maximal (≤32 words) "
                 "shrinks both the gaps and the slopes", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
