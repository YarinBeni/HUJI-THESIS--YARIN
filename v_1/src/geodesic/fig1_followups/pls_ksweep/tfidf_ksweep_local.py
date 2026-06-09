#!/usr/bin/env python3
"""Lean LOCAL TF-IDF k-sweep — the 4th curve, computed on the laptop.

The cluster backfill timed out because run_mc_probes.run_tfidf_pls does ~6x the
needed work (both cleanings x raw+log year x ruler PLS-DA x shuffled nulls).
TF-IDF is text-only, so we compute exactly what the figures need here: balanced
tier0 year-raw Spearman per k, over the 200 MC draws, parallel over draws.

Then it MERGES tfidf into pls_components_tradeoff.csv + best_k_vs_fixed_k.csv and
regenerates pls_components_tradeoff.png, best_k_vs_fixed_k.png, per_method_panels.png.

Uses fit_pls_groupkfold (the same selector the neural curves used) so tfidf is
apples-to-apples. Run from repo root: python .../tfidf_ksweep_local.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[5]
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing"))
from pls_utils import fit_pls_groupkfold          # noqa: E402

KS = [1, 2, 3, 5, 8, 16, 32, 64]                  # 128 always exceeds the balanced cap
RIDGE_TFIDF = 0.355                                # T2 balanced tier0 year-raw best (L00)
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 5))
PK = _REPO / "v_1/src/geodesic/fig1_followups/pls_ksweep"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"


def _draw_sweep(di, draws, df, frag_order):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    row = draws[di]
    pos = np.where(row)[0] if row.dtype == bool else row.astype(int)
    sub = df.iloc[pos]
    X = normalize(TfidfVectorizer(**TFIDF).fit_transform(sub["text_tier0"].fillna("").astype(str)),
                  norm="l2").toarray().astype(np.float32)
    y = sub["year"].astype(float).values
    g = sub["ruler"].astype(str).values
    out = {}
    for k in KS:
        try:
            out[k] = fit_pls_groupkfold(X, y, g, n_components=k, n_splits=5)["spearman_mean"]
        except Exception:
            out[k] = np.nan
    return out


def merge_csv(path: Path, rows_new, key_prefix):
    """Drop existing tfidf rows, append new ones, keep header."""
    existing = list(csv.reader(open(path)))
    head, body = existing[0], [r for r in existing[1:] if r and r[0] != "tfidf"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(body); w.writerows(rows_new)


def main():
    draws = np.load(BAL / "draws_matrix.npy")
    frag_order = __import__("json").load(open(BAL / "corpus_fragment_order.json"))
    df = pd.read_parquet(_REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet").reset_index(drop=True)
    n = draws.shape[0]
    print(f"[tfidf] sweeping k={KS} over {n} balanced draws (parallel)...")
    per_draw = Parallel(n_jobs=-1, verbose=5)(
        delayed(_draw_sweep)(di, draws, df, frag_order) for di in range(n))

    # per-k mean/std  -> tradeoff rows
    tradeoff_rows = []
    for k in KS:
        vals = [d[k] for d in per_draw if np.isfinite(d[k])]
        if vals:
            tradeoff_rows.append(["tfidf", k, np.mean(vals), np.std(vals), RIDGE_TFIDF])
    merge_csv(PK / "pls_components_tradeoff.csv", tradeoff_rows, "tfidf")

    # best-k row
    fixed3 = np.nanmean([d[3] for d in per_draw])
    bestk = np.nanmean([max(v for v in d.values() if np.isfinite(v)) for d in per_draw])
    merge_csv(PK / "best_k_vs_fixed_k.csv",
              [["tfidf", n, fixed3, bestk, RIDGE_TFIDF, bestk - fixed3]], "tfidf")
    print(f"[tfidf] fixed_k3={fixed3:.3f} best_k_per_draw={bestk:.3f} ridge={RIDGE_TFIDF} "
          f"peak_fixed={max(r[2] for r in tradeoff_rows):.3f}")

    # regenerate the three figures from the merged CSVs
    _replot_tradeoff(); _replot_bestk()
    subprocess.run([sys.executable, str(PK / "per_method_panels.py"),
                    "--tradeoff-csv", str(PK / "pls_components_tradeoff.csv"),
                    "--bestk-csv", str(PK / "best_k_vs_fixed_k.csv")], check=True)
    print("[ok] regenerated tradeoff / best_k / per_method_panels with the tfidf curve")


def _style(plt):
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def _replot_tradeoff():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    rows = list(csv.DictReader(open(PK / "pls_components_tradeoff.csv")))
    models = sorted({r["model"] for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, c in zip(models, colors):
        rs = sorted((r for r in rows if r["model"] == m), key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in rs]; mu = [float(r["spearman_mean"]) for r in rs]
        sd = [float(r["spearman_std"]) for r in rs]
        ax.plot(ks, mu, "-o", color=c, label=m)
        ax.fill_between(ks, np.array(mu) - sd, np.array(mu) + sd, color=c, alpha=0.12)
        rb = rs[0].get("ridge_baseline")
        if rb not in (None, "", "None"):
            ax.axhline(float(rb), color=c, ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log", base=2); ax.set_xlabel("PLS components k (log2)")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD)")
    ax.set_title("PLS components tradeoff — dashed = Ridge (all columns)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(PK / "pls_components_tradeoff.png"); plt.close(fig)


def _replot_bestk():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    rows = list(csv.DictReader(open(PK / "best_k_vs_fixed_k.csv")))
    models = [r["model"] for r in rows]
    fixed = [float(r["fixed_k3"]) for r in rows]
    best = [float(r["best_k_per_draw"]) for r in rows]
    ridge = [float(r["ridge"]) for r in rows]
    x = np.arange(len(models)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 2, 5))
    ax.bar(x - w, fixed, w, color="#3b6ea8", label="PLS fixed k=3 (honest)")
    ax.bar(x, best, w, color="#f0a202", label="PLS best-k-per-draw (Fig-1A)")
    ax.bar(x + w, ridge, w, color="#c0504d", label="Ridge (all columns)")
    for i in range(len(models)):
        for off, v in [(-w, fixed[i]), (0, best[i]), (w, ridge[i])]:
            ax.text(i + off, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD)")
    ax.set_title("best-k-per-draw vs fixed k=3 vs Ridge (orange − blue = selection inflation)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(PK / "best_k_vs_fixed_k.png"); plt.close(fig)


if __name__ == "__main__":
    main()
