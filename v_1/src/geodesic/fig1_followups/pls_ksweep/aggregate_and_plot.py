#!/usr/bin/env python3
"""Task 4 — PLS components tradeoff plot (LOCAL, runs after the cluster job).

Reads the per-draw JSONs written by run_mc_probes.py with
--method-tag mc_balanced_ksweep, extracts year-Spearman at each PLS n_components
k for the 4 Fig-1A models (at their best balanced layer), averages over the 200
draws, and plots k vs Spearman (mean ± SD). Ridge is drawn as a horizontal
reference line per model (the "use all columns" baseline) from T2_year_ridge.csv.

Usage:
    python aggregate_and_plot.py \
        --probes-dir  v_1/src/geodesic/fig1_followups/pls_ksweep/probes \
        --t2-csv      v_1/src/geodesic/results/tables/T2_year_ridge.csv \
        --out         v_1/src/geodesic/fig1_followups/pls_ksweep/pls_components_tradeoff.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# probe name -> (config_key for its best balanced layer, year=raw)
MODELS = {
    "mlm_pls":                 ("mlm",                 "mlm__tier0__mean__L01__year-raw"),
    "tfidf_pls":               ("tfidf",               "tfidf__tier0__na__L00__year-raw"),
    "thalesian_cunei400m_pls": ("thalesian_cunei400m", "thalesian_cunei400m__tier0__mean__L12__year-raw"),
    "qwen3_32b_pls":           ("qwen3_32b",           "qwen3_32b__tier0__mean__L09__year-raw"),
}
METHOD_TAG = "mc_balanced_ksweep"


def collect(probes_dir: Path, probe: str, cfg_key: str) -> dict[int, list[float]]:
    """k -> [spearman_mean per draw]."""
    per_k: dict[int, list[float]] = defaultdict(list)
    files = sorted(probes_dir.glob(f"{probe}__{METHOD_TAG}__draw*.json"))
    for fp in files:
        try:
            doc = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        rec = doc.get("results", {}).get(cfg_key)
        if rec is None or "metrics_per_k" not in rec:
            continue
        for k_str, m in rec["metrics_per_k"].items():
            v = m.get("spearman_mean")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                per_k[int(k_str)].append(float(v))
    return per_k


def ridge_baseline(t2_csv: Path, model: str) -> float | None:
    """Best balanced year-raw Spearman for `model` from T2 (Ridge)."""
    if not t2_csv.exists():
        return None
    best = None
    for r in csv.DictReader(open(t2_csv)):
        if (r.get("model") == model and r.get("regime") == "balanced"
                and r.get("year_transform") == "raw"):
            try:
                v = float(r["spearman_mean"])
            except (ValueError, KeyError):
                continue
            best = v if best is None else max(best, v)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", type=Path, required=True)
    ap.add_argument("--t2-csv", type=Path,
                    default=Path("v_1/src/geodesic/results/tables/T2_year_ridge.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("v_1/src/geodesic/fig1_followups/pls_ksweep/pls_components_tradeoff.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

    summary_rows = []
    for (probe, (model, cfg_key)), c in zip(MODELS.items(), colors):
        per_k = collect(args.probes_dir, probe, cfg_key)
        if not per_k:
            print(f"[warn] no data for {model} ({cfg_key})")
            continue
        ks = sorted(per_k)
        means = np.array([np.mean(per_k[k]) for k in ks])
        stds = np.array([np.std(per_k[k]) for k in ks])
        ndraws = len(per_k[ks[0]])
        ax.plot(ks, means, "-o", color=c, label=f"{model} (PLS, n={ndraws})")
        ax.fill_between(ks, means - stds, means + stds, color=c, alpha=0.15)

        rb = ridge_baseline(args.t2_csv, model)
        if rb is not None:
            ax.axhline(rb, color=c, ls="--", lw=1, alpha=0.7)
        for k in ks:
            summary_rows.append((model, k, float(np.mean(per_k[k])),
                                 float(np.std(per_k[k])), rb))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("PLS components (k)  [log2]")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD over draws)")
    ax.set_title("PLS components tradeoff — dashed = Ridge (all columns)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[ok] wrote {args.out}")

    csv_out = args.out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "k", "spearman_mean", "spearman_std", "ridge_baseline"])
        w.writerows(summary_rows)
    print(f"[ok] wrote {csv_out}")


if __name__ == "__main__":
    main()
