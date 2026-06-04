#!/usr/bin/env python3
"""Reproduce Fig-1A's best-k-per-draw PLS and compare to fixed-k + Ridge.

Reads the raw per-draw sweep JSONs (the ones only on the cluster) and computes,
per model at its best balanced layer:

  - fixed-k       : mean over draws of Spearman at a single fixed k (honest,
                    matched to Ridge's single knob)
  - best-k-per-draw : for EACH draw take its own best-k Spearman, then average.
                    This is exactly run_mc_probes.py's `best_k_by_spearman`
                    estimator (run_mc_probes.py:511) that feeds T1 / Fig-1A.
  - ridge         : the all-columns baseline (from the tradeoff CSV).

If best-k-per-draw lands back near the T1/Fig-1A number while fixed-k sits at
the tradeoff peak, the Fig-1A "PLS > Ridge" gap is the k-pick, full stop.

Usage:
    python best_k_compare.py \
        --probes-dir .../pls_ksweep/probes \
        --tradeoff-csv .../pls_ksweep/pls_components_tradeoff.csv \
        --fixed-k 3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# probe -> (model, best-layer cfg key, year=raw)
MODELS = {
    "mlm_pls":                 ("mlm",                 "mlm__tier0__mean__L01__year-raw"),
    "tfidf_pls":               ("tfidf",               "tfidf__tier0__na__L00__year-raw"),
    "thalesian_cunei400m_pls": ("thalesian_cunei400m", "thalesian_cunei400m__tier0__mean__L12__year-raw"),
    "qwen3_32b_pls":           ("qwen3_32b",           "qwen3_32b__tier0__mean__L09__year-raw"),
}
METHOD_TAG = "mc_balanced_ksweep"


def per_draw_sp(probes_dir: Path, probe: str, cfg_key: str):
    """Return list of per-draw {k: spearman_mean} dicts for one model."""
    out = []
    for fp in sorted(probes_dir.glob(f"{probe}__{METHOD_TAG}__draw*.json")):
        try:
            doc = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        rec = doc.get("results", {}).get(cfg_key)
        if rec and "metrics_per_k" in rec:
            d = {int(k): m.get("spearman_mean") for k, m in rec["metrics_per_k"].items()
                 if m.get("spearman_mean") is not None
                 and not (isinstance(m.get("spearman_mean"), float) and np.isnan(m["spearman_mean"]))}
            if d:
                out.append(d)
    return out


def ridge_from_csv(csv_path: Path, model: str):
    if not csv_path.exists():
        return None
    for r in csv.DictReader(open(csv_path)):
        if r["model"] == model and r.get("ridge_baseline") not in ("", "None", None):
            return float(r["ridge_baseline"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", type=Path, required=True)
    ap.add_argument("--tradeoff-csv", type=Path,
                    default=Path("v_1/src/geodesic/fig1_followups/pls_ksweep/pls_components_tradeoff.csv"))
    ap.add_argument("--fixed-k", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("v_1/src/geodesic/fig1_followups/pls_ksweep/best_k_vs_fixed_k.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, fixed, fixed_sd, bestk, bestk_sd, ridge = [], [], [], [], [], []
    rows_out = []
    for probe, (model, cfg) in MODELS.items():
        draws = per_draw_sp(args.probes_dir, probe, cfg)
        if not draws:
            print(f"[warn] no draws for {model}")
            continue
        fk = [d[args.fixed_k] for d in draws if args.fixed_k in d]
        bk = [max(d.values()) for d in draws]                     # per-draw best k
        rb = ridge_from_csv(args.tradeoff_csv, model)
        names.append(model)
        fixed.append(np.mean(fk)); fixed_sd.append(np.std(fk))
        bestk.append(np.mean(bk)); bestk_sd.append(np.std(bk))
        ridge.append(rb if rb is not None else np.nan)
        rows_out.append((model, len(draws), np.mean(fk), np.mean(bk), rb,
                         np.mean(bk) - np.mean(fk)))
        print(f"  {model:22s} n={len(draws):3d}  fixed-k{args.fixed_k}={np.mean(fk):.3f}  "
              f"best-k-per-draw={np.mean(bk):.3f}  ridge={rb}  "
              f"selection_gap=+{np.mean(bk)-np.mean(fk):.3f}")

    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2, 5))
    ax.bar(x - w, fixed, w, yerr=fixed_sd, capsize=3, color="#3b6ea8",
           label=f"PLS fixed k={args.fixed_k} (honest)")
    ax.bar(x, bestk, w, yerr=bestk_sd, capsize=3, color="#f0a202",
           label="PLS best-k-per-draw (Fig-1A)")
    ax.bar(x + w, ridge, w, color="#c0504d", label="Ridge (all columns)")
    for i in range(len(names)):
        ax.text(i - w, fixed[i], f"{fixed[i]:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i, bestk[i], f"{bestk[i]:.3f}", ha="center", va="bottom", fontsize=7)
        if not np.isnan(ridge[i]):
            ax.text(i + w, ridge[i], f"{ridge[i]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD)")
    ax.set_title(f"The k-pick, isolated: best-k-per-draw vs fixed k={args.fixed_k} vs Ridge\n"
                 "(orange − blue = the Fig-1A selection inflation)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"[ok] wrote {args.out}")

    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "n_draws", f"fixed_k{args.fixed_k}", "best_k_per_draw",
                     "ridge", "selection_gap"])
        wr.writerows(rows_out)
    print(f"[ok] wrote {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
