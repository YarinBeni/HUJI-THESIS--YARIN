#!/usr/bin/env python3
"""Map the error landscape: where each model fails, and what makes a fragment
fail for everyone. Joins predictions with the corpus text (length + tokens).

Prints the diagnostics (length-dependence, failure-bucket profiles, tokens
enriched in the all-wrong set, prediction-bias-when-wrong) and writes
error_map_summary.png (length-dependence per model + universal failure drivers).

Usage:
    python error_map.py --pred-csv .../predictions.csv \
        --corpus v_1/data/evaluation/corpora/orcc_corpus.parquet --tol 100
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["mlm", "thalesian_cunei400m", "qwen3_32b", "tfidf"]
PAL = {"mlm": "#0072B2", "thalesian_cunei400m": "#E69F00",
       "qwen3_32b": "#009E73", "tfidf": "#D55E00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--corpus", type=Path,
                    default=Path("v_1/data/evaluation/corpora/orcc_corpus.parquet"))
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.pred_csv.parent / "error_map_summary.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#dddddd", "axes.axisbelow": True, "font.size": 11,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False})

    pred = pd.read_csv(args.pred_csv)
    corp = pd.read_parquet(args.corpus)[["fragment_id", "word_count", "text_tier0"]]
    df = pred.merge(corp, on="fragment_id", how="left")
    for m in MODELS:
        df[f"ok_{m}"] = (df[f"pred_{m}"] - df["year_true"]).abs() <= args.tol
    df["n_correct"] = df[[f"ok_{m}" for m in MODELS]].sum(axis=1)

    # diagnostics -----------------------------------------------------------
    df["wcq"] = pd.qcut(df.word_count, 4, labels=False)
    print("=== accuracy by length quartile ===")
    for q in range(4):
        g = df[df.wcq == q]
        print(f"  Q{q+1} (~{int(g.word_count.median())}w, n={len(g)}): " +
              "  ".join(f"{m.split('_')[0]}={g[f'ok_{m}'].mean():.2f}" for m in MODELS))

    broken = df.text_tier0.fillna("").str.contains("he-pi2|eš-šu2")
    df["_broken"], df["_short"], df["_rare"] = broken, df.word_count < 15, df.period != "Neo-Assyrian"
    aw = df[df.n_correct == 0]
    print(f"\n=== all-4-wrong set (n={len(aw)}) drivers vs corpus ===")
    for d, nm in [("_broken", "damaged"), ("_short", "short<15w"), ("_rare", "rare-period")]:
        print(f"  {nm:12s} all-wrong={aw[d].mean():.0%}  corpus={df[d].mean():.0%}")

    allc = Counter(); hardc = Counter()
    for t in df.text_tier0.dropna():
        allc.update(str(t).split())
    for t in aw.text_tier0.dropna():
        hardc.update(str(t).split())
    na, nh = sum(allc.values()), sum(hardc.values())
    print("\n=== tokens enriched in all-wrong ===")
    enr = sorted(((w, hardc[w] / nh / (allc[w] / na + 1e-9), hardc[w])
                  for w in hardc if hardc[w] >= 5), key=lambda x: -x[1])
    for w, r, c in enr[:10]:
        print(f"  {w:20s} x{r:5.1f}  (n={c})")

    # plot ------------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    qmed = [int(df[df.wcq == q].word_count.median()) for q in range(4)]
    for m in MODELS:
        axA.plot(range(4), [df[df.wcq == q][f"ok_{m}"].mean() for q in range(4)],
                 "-o", lw=2.4, ms=7, color=PAL[m], label=m)
    axA.set_xticks(range(4)); axA.set_xticklabels([f"Q{q+1}\n~{qmed[q]}w" for q in range(4)])
    axA.set_xlabel("text length (word-count quartile)")
    axA.set_ylabel(f"fraction correct (±{args.tol:.0f} yr)")
    axA.set_ylim(0.5, 1.0); axA.set_title("Length is the main axis of failure")
    axA.legend(loc="lower right", fontsize=9)

    drivers = ["_broken", "_short", "_rare"]
    names = ["damaged\n(he-pi2/eššu)", "very short\n(<15 words)", "rare period\n(not Neo-Assyrian)"]
    x = np.arange(3); w = 0.38
    axB.bar(x - w / 2, [df[d].mean() for d in drivers], w, color="#bbbbbb", label="whole corpus")
    axB.bar(x + w / 2, [aw[d].mean() for d in drivers], w, color="#8B0000",
            label=f"all-4-wrong (n={len(aw)})")
    for i, d in enumerate(drivers):
        axB.text(i - w / 2, df[d].mean(), f"{df[d].mean():.0%}", ha="center", va="bottom", fontsize=8)
        axB.text(i + w / 2, aw[d].mean(), f"{aw[d].mean():.0%}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(names, fontsize=9); axB.set_ylim(0, 1)
    axB.set_ylabel("fraction of fragments"); axB.set_title("What makes a fragment fail for EVERY model")
    axB.legend(fontsize=9); axB.grid(axis="x", visible=False)
    fig.suptitle("Where dating fails: per-model length-dependence + universal failure drivers",
                 y=1.02, fontsize=13)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
