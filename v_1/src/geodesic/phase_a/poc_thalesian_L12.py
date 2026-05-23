#!/usr/bin/env python3
"""Phase A — Single-layer proof of concept: Thalesian cuneiBase-400m, layer 12,
tier0/mean (Round 2 best configuration).

Gate: proceed to Phase B if either Isomap or earliest-bin geodesic improves
Spearman by >=0.05 over PLS, OR pairwise-order accuracy >=0.70.
Stop and report null if both underperform PLS by >=0.05 AND pairwise acc <0.60.

Outputs:
  v_1/src/geodesic/results/phase_a_results.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "v_1/src/geodesic"))

from utils import (
    available_layers,
    earliest_bin_coord,
    find_acts_dir,
    geodesic_dist,
    build_knn_graph,
    isomap_1d,
    load_layer,
    load_year_labels,
    neighbor_purity,
    pairwise_order_acc_fast,
    pls_pairwise_acc,
    pca_l2,
    sign_flip_coord,
)

# ── config ─────────────────────────────────────────────────────────────────
METHOD   = "thalesian_cunei400m"
CLEANING = "tier0"
POOL     = "mean"
LAYER    = 12
N_PCA    = 64
MARGIN   = 100   # years
PARQUET  = ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_JSON = ROOT / "v_1/src/geodesic/results/phase_a_results.json"

# PLS Round 2 reference numbers (from pls_best_layers.json)
PLS_SPEARMAN_REF = 0.4670
PLS_R2_REF       = 0.1055
PLS_MAE_REF      = 75.1


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ── load activations ───────────────────────────────────────────────────
    acts_dir = find_acts_dir(METHOD, CLEANING, POOL, repo_root=ROOT)
    if acts_dir is None:
        sys.exit(f"ERROR: activations not found for {METHOD}/{CLEANING}/{POOL}")
    print(f"Activations: {acts_dir}")
    print(f"Available layers: {available_layers(acts_dir)}")

    X_full = load_layer(acts_dir, LAYER)
    print(f"X_full shape: {X_full.shape}")

    # ── align with year-labeled rows ───────────────────────────────────────
    frag_indices, years = load_year_labels(PARQUET)
    X = X_full[frag_indices]   # (n_labeled, d_model)
    n = len(years)
    print(f"Year-labeled fragments: {n}  (d_model={X.shape[1]})")

    # ── preprocessing ──────────────────────────────────────────────────────
    X_pca = pca_l2(X, n_components=N_PCA)
    print(f"PCA shape: {X_pca.shape}")

    # ── build kNN graph ────────────────────────────────────────────────────
    k_used, adj = build_knn_graph(X_pca, k_min=3, k_max=50, metric="cosine")
    print(f"kNN graph connected at k={k_used}")

    # ── compute geodesic distance matrix ──────────────────────────────────
    print("Computing geodesic distances...")
    dist = geodesic_dist(adj)
    n_inf = np.isinf(dist).sum()
    if n_inf > 0:
        print(f"WARNING: {n_inf} inf entries in dist matrix — graph not fully connected")
        dist = np.where(np.isinf(dist), dist[~np.isinf(dist)].max() * 2, dist)

    # ── coordinate A1: Isomap 1D ───────────────────────────────────────────
    print("Fitting Isomap...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord_isomap = isomap_1d(X_pca, k=k_used, metric="cosine")
    coord_isomap = sign_flip_coord(coord_isomap, years)

    # ── coordinate A2: earliest-bin geodesic ──────────────────────────────
    coord_ebin = earliest_bin_coord(dist, years, bin_width=MARGIN)
    coord_ebin = sign_flip_coord(coord_ebin, years)

    # ── metrics ────────────────────────────────────────────────────────────
    print("Computing metrics (this may take ~1 min for pairwise acc)...")

    def metrics(coord, name):
        r, _ = spearmanr(coord, years)
        acc = pairwise_order_acc_fast(coord, years, margin=MARGIN)
        purity, null_mean, null_std = neighbor_purity(coord, years, k=10,
                                                       window=MARGIN, n_perm=500)
        sigma_above = (purity - null_mean) / null_std if null_std > 0 else float("nan")
        print(f"  [{name}] Spearman={r:.4f}  PairwiseAcc={acc:.4f}  "
              f"Purity={purity:.4f} ({sigma_above:.1f}σ above null)")
        return {
            "spearman": round(float(r), 4),
            "pairwise_order_acc": round(float(acc), 4),
            "neighbor_purity": round(float(purity), 4),
            "null_purity_mean": round(float(null_mean), 4),
            "null_purity_std": round(float(null_std), 4),
            "sigma_above_null": round(float(sigma_above), 2),
        }

    print("\n--- Geodesic coordinates ---")
    m_isomap = metrics(coord_isomap, "A1-Isomap")
    m_ebin   = metrics(coord_ebin,   "A2-EarliestBin")

    # ── PLS baseline ───────────────────────────────────────────────────────
    print("\n--- PLS baseline (refit on layer 12) ---")
    pls_acc = pls_pairwise_acc(X_pca, years, margin=MARGIN)
    pls_spearman_refit, _ = spearmanr(
        __import__("sklearn.cross_decomposition", fromlist=["PLSRegression"])
        .PLSRegression(n_components=1).fit(X_pca, years).predict(X_pca).ravel(),
        years
    )
    print(f"  [PLS-refit] Spearman={pls_spearman_refit:.4f}  PairwiseAcc={pls_acc:.4f}")
    print(f"  [PLS-R2ref] Spearman={PLS_SPEARMAN_REF}  (from pls_best_layers.json)")

    # ── gate evaluation ────────────────────────────────────────────────────
    best_spearman = max(m_isomap["spearman"], m_ebin["spearman"])
    best_acc      = max(m_isomap["pairwise_order_acc"], m_ebin["pairwise_order_acc"])
    delta_spearman = best_spearman - PLS_SPEARMAN_REF

    gate_pass = (delta_spearman >= 0.05) or (best_acc >= 0.70)
    gate_null = (delta_spearman <= -0.05) and (best_acc < 0.60)

    if gate_pass:
        verdict = "PASS → proceed to Phase B"
    elif gate_null:
        verdict = "NULL → round 3 negative result; proceed to Phase 2 (scale)"
    else:
        verdict = "MARGINAL → discuss with advisor before proceeding"

    print(f"\n=== Phase A Gate ===")
    print(f"  Best geodesic Spearman={best_spearman:.4f}  delta vs PLS={delta_spearman:+.4f}")
    print(f"  Best pairwise-order acc={best_acc:.4f}")
    print(f"  Verdict: {verdict}")

    # ── save results ───────────────────────────────────────────────────────
    results = {
        "config": {
            "method": METHOD, "cleaning": CLEANING, "pool": POOL,
            "layer": LAYER, "n_pca": N_PCA, "margin": MARGIN,
            "k_used": k_used, "n_fragments": n,
        },
        "pls_reference": {
            "spearman_from_pls_best_layers": PLS_SPEARMAN_REF,
            "r2_from_pls_best_layers": PLS_R2_REF,
            "mae_from_pls_best_layers": PLS_MAE_REF,
            "spearman_refit": round(float(pls_spearman_refit), 4),
            "pairwise_order_acc_refit": round(float(pls_acc), 4),
        },
        "isomap": m_isomap,
        "earliest_bin_geodesic": m_ebin,
        "gate": {
            "best_spearman": round(float(best_spearman), 4),
            "delta_vs_pls": round(float(delta_spearman), 4),
            "best_pairwise_acc": round(float(best_acc), 4),
            "pass": gate_pass,
            "null": gate_null,
            "verdict": verdict,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()
