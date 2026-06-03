#!/usr/bin/env python3
"""Task 3 — error-overlap analysis (LOCAL, runs after the cluster dump).

Reads predictions.csv (one OOF predicted year per fragment per model) and asks:
do the 4 models get the same fragments right/wrong, or different ones?

Outputs (under --out-dir):
  - error_correlation.png   Spearman corr of per-fragment |error| across models
  - correct_jaccard.png     Jaccard overlap of "within ±tol yr" correct-sets
  - n_models_correct.png    histogram: how many models date each fragment well
  - overlap_summary.csv      the numbers behind the plots

High error-correlation + high Jaccard => models share one surface signal
(supports "dating is shallow"). Low => something model-specific.

Usage:
    python analyze_overlap.py \
        --pred-csv v_1/src/geodesic/fig1_followups/error_overlap/predictions/predictions.csv \
        --tol 100
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load(pred_csv: Path):
    rows = list(csv.DictReader(open(pred_csv)))
    models = [c[len("pred_"):] for c in rows[0] if c.startswith("pred_")]
    # metadata slice labels = any column that is neither id/target nor a pred_*
    labels = [c for c in rows[0]
              if c not in ("fragment_id", "year_true") and not c.startswith("pred_")]
    y_true = np.array([float(r["year_true"]) for r in rows])
    preds, mask = {}, {}
    for m in models:
        vals = np.array([float(r[f"pred_{m}"]) if r[f"pred_{m}"] not in ("", "nan") else np.nan
                         for r in rows])
        preds[m] = vals
        mask[m] = ~np.isnan(vals)
    meta = {lab: np.array([r[lab] for r in rows]) for lab in labels}
    return models, y_true, preds, mask, meta


def breakdown_by_label(label, values, models, correct, mask, out_dir, plt,
                       min_n=15, top_k=20):
    """Heatmap of fraction-correct per (group × model) for one metadata label,
    so you can see whether shared errors cluster on a ruler / period / etc."""
    groups, counts = np.unique(values, return_counts=True)
    keep = [g for g, c in sorted(zip(groups, counts), key=lambda x: -x[1])
            if c >= min_n][:top_k]
    if not keep:
        print(f"[skip] {label}: no group with >= {min_n} fragments")
        return
    M = np.full((len(keep), len(models)), np.nan)
    for gi, g in enumerate(keep):
        in_g = values == g
        for mi, m in enumerate(models):
            sel = in_g & mask[m]
            if sel.sum():
                M[gi, mi] = correct[m][sel].mean()
    fig, ax = plt.subplots(figsize=(1.4 * len(models) + 3, 0.45 * len(keep) + 2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([f"{g} (n={int((values == g).sum())})" for g in keep], fontsize=8)
    for gi in range(len(keep)):
        for mi in range(len(models)):
            if not np.isnan(M[gi, mi]):
                ax.text(mi, gi, f"{M[gi, mi]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Fraction dated within tol, by {label}")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_dir / f"by_{label}.png", dpi=150); plt.close(fig)
    print(f"[ok] by_{label}.png ({len(keep)} groups)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=100.0,
                    help="abs-error threshold (yr) for 'correct' (default 100)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=15,
                    help="min fragments for a metadata group to be plotted (default 15)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="max groups per label, by frequency (default 20)")
    args = ap.parse_args()
    out_dir = args.out_dir or args.pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models, y_true, preds, mask, meta = load(args.pred_csv)
    n = len(models)
    abs_err = {m: np.abs(preds[m] - y_true) for m in models}
    correct = {m: (abs_err[m] <= args.tol) & mask[m] for m in models}

    # 1) error correlation (Spearman of |error|, over fragments both models predicted)
    corr = np.eye(n)
    for i, j in combinations(range(n), 2):
        both = mask[models[i]] & mask[models[j]]
        rho = spearmanr(abs_err[models[i]][both], abs_err[models[j]][both]).statistic
        corr[i, j] = corr[j, i] = rho

    # 2) Jaccard of correct-sets
    jac = np.eye(n)
    for i, j in combinations(range(n), 2):
        a, b = correct[models[i]], correct[models[j]]
        inter = np.sum(a & b)
        union = np.sum(a | b)
        jac[i, j] = jac[j, i] = (inter / union) if union else 0.0

    # 3) how many models correct per fragment (over fragments all models predicted)
    all_pred = np.all([mask[m] for m in models], axis=0)
    n_correct = np.sum([correct[m][all_pred] for m in models], axis=0)
    hist = np.bincount(n_correct.astype(int), minlength=n + 1)

    def heatmap(M, title, fname, fmt="{:.2f}"):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=8)
        ax.set_title(title); fig.colorbar(im); fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150); plt.close(fig)
        print(f"[ok] {fname}")

    heatmap(corr, "Per-fragment |error| Spearman corr", "error_correlation.png")
    heatmap(jac, f"Correct-set Jaccard (±{args.tol:.0f} yr)", "correct_jaccard.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(n + 1), hist, color="steelblue")
    ax.set_xlabel("# models that date the fragment within ±%g yr" % args.tol)
    ax.set_ylabel(f"# fragments (of {int(all_pred.sum())})")
    ax.set_title("Agreement: shared-easy vs model-specific fragments")
    for x, h in enumerate(hist):
        ax.text(x, h, str(int(h)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "n_models_correct.png", dpi=150); plt.close(fig)
    print("[ok] n_models_correct.png")

    # per-metadata-label breakdown: where do the shared errors cluster?
    for label, values in meta.items():
        breakdown_by_label(label, values, models, correct, mask, out_dir, plt,
                            min_n=args.min_n, top_k=args.top_k)

    with open(out_dir / "overlap_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + models)
        for i, m in enumerate(models):
            w.writerow([f"err_corr_vs[{m}]"] + [f"{corr[i, j]:.3f}" for j in range(n)])
        for i, m in enumerate(models):
            w.writerow([f"jaccard_vs[{m}]"] + [f"{jac[i, j]:.3f}" for j in range(n)])
        w.writerow([])
        w.writerow(["n_models_correct"] + [str(k) for k in range(n + 1)])
        w.writerow(["n_fragments"] + [str(int(h)) for h in hist])
        for m in models:
            w.writerow([f"frac_correct[{m}]", f"{np.mean(correct[m][mask[m]]):.3f}"])
    print(f"[ok] overlap_summary.csv  (tol=±{args.tol:.0f} yr, {len(models)} models)")


if __name__ == "__main__":
    main()
