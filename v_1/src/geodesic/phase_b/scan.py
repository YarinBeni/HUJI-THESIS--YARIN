#!/usr/bin/env python3
"""Phase B: Full layer × method geodesic scoreboard.

Runs the geodesic pipeline (PCA64 → L2 → kNN → Isomap 1D + earliest-bin)
for every available layer of one (method, cleaning, pool) combo.
Results are written incrementally (crash-safe).

Usage:
    python v_1/src/geodesic/phase_b/scan.py \
        --method thalesian_cunei400m --cleaning tier0 --pool mean
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr as _spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "v_1/src/geodesic"))

from utils import (
    find_acts_dir,
    load_layer,
    load_year_labels,
    available_layers,
    pca_l2,
    build_knn_graph,
    isomap_1d,
    geodesic_dist,
    earliest_bin_coord,
    sign_flip_coord,
    pairwise_order_acc_fast,
    neighbor_purity,
)


def _sp(a, b) -> float:
    res = _spearmanr(a, b)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def run_layer(X_raw: np.ndarray, frag_idx: np.ndarray, years: np.ndarray,
              n_perm: int = 50) -> dict:
    X = X_raw[frag_idx]
    result = {}

    try:
        X_pca = pca_l2(X, n_components=64)
    except Exception as e:
        return {"error": f"pca_l2: {e}"}

    try:
        k, graph = build_knn_graph(X_pca, metric="cosine")
        result["k_used"] = int(k)
    except Exception as e:
        return {"error": f"build_knn_graph: {e}"}

    # Isomap 1D
    try:
        coord = isomap_1d(X_pca, k, metric="cosine")
        coord = sign_flip_coord(coord, years)
        sp = _sp(coord, years)
        pacc = pairwise_order_acc_fast(coord, years, margin=100)
        purity, null_mean, null_std = neighbor_purity(coord, years, n_perm=n_perm)
        sigma = (purity - null_mean) / null_std if null_std > 0 else float("nan")
        result["isomap"] = {
            "spearman": sp,
            "pairwise_order_acc": pacc,
            "neighbor_purity": purity,
            "neighbor_purity_null_mean": null_mean,
            "neighbor_purity_sigma": sigma,
        }
    except Exception as e:
        result["isomap"] = {"error": str(e)}

    # Earliest-bin geodesic
    try:
        dist = geodesic_dist(graph)
        coord_e = earliest_bin_coord(dist, years, bin_width=100)
        coord_e = sign_flip_coord(coord_e, years)
        sp_e = _sp(coord_e, years)
        pacc_e = pairwise_order_acc_fast(coord_e, years, margin=100)
        purity_e, null_mean_e, null_std_e = neighbor_purity(coord_e, years, n_perm=n_perm)
        sigma_e = (purity_e - null_mean_e) / null_std_e if null_std_e > 0 else float("nan")
        result["earliest_bin"] = {
            "spearman": sp_e,
            "pairwise_order_acc": pacc_e,
            "neighbor_purity": purity_e,
            "neighbor_purity_null_mean": null_mean_e,
            "neighbor_purity_sigma": sigma_e,
        }
    except Exception as e:
        result["earliest_bin"] = {"error": str(e)}

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",      required=True)
    ap.add_argument("--cleaning",    required=True)
    ap.add_argument("--pool",        required=True)
    ap.add_argument("--output-dir",  default="v_1/src/geodesic/results/phase_b")
    ap.add_argument("--parquet",     default="v_1/data/evaluation/corpora/orcc_corpus.parquet")
    ap.add_argument("--n-perm",      type=int, default=50,
                    help="Permutations for neighbor-purity null (50 for speed in B; 500 for final)")
    args = ap.parse_args()

    acts_dir = find_acts_dir(args.method, args.cleaning, args.pool)
    if acts_dir is None:
        print(f"SKIP: activations not found for {args.method}/{args.cleaning}/{args.pool}")
        sys.exit(0)

    print(f"Acts dir : {acts_dir}")

    parquet = ROOT / args.parquet
    frag_idx, years = load_year_labels(parquet)
    print(f"Labeled fragments: {len(frag_idx)}  year range: {years.min():.0f}–{years.max():.0f}")

    layers = available_layers(acts_dir)
    print(f"Layers   : {layers}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"phase_b_{args.method}_{args.cleaning}_{args.pool}.json"

    existing = {}
    if out_file.exists():
        existing = json.loads(out_file.read_text()).get("layers", {})
        print(f"Resuming: {len(existing)} layers already done")

    layer_results = dict(existing)

    for layer in layers:
        key = str(layer)
        if key in layer_results:
            print(f"  L{layer:02d}: skip (cached)")
            continue

        print(f"  L{layer:02d}: loading ...", flush=True)
        try:
            X_raw = load_layer(acts_dir, layer)
        except Exception as e:
            layer_results[key] = {"error": str(e)}
            continue

        print(f"  L{layer:02d}: n={len(frag_idx)} d={X_raw.shape[1]} — running ...", flush=True)
        res = run_layer(X_raw, frag_idx, years, n_perm=args.n_perm)
        layer_results[key] = res

        iso = res.get("isomap", {})
        print(f"  L{layer:02d}: isomap sp={iso.get('spearman', float('nan')):.3f}  "
              f"pacc={iso.get('pairwise_order_acc', float('nan')):.3f}  "
              f"purity_σ={iso.get('neighbor_purity_sigma', float('nan')):.1f}", flush=True)

        out_file.write_text(json.dumps({
            "method":   args.method,
            "cleaning": args.cleaning,
            "pool":     args.pool,
            "n_perm":   args.n_perm,
            "layers":   layer_results,
        }, indent=2))

    print(f"\nDone → {out_file}")


if __name__ == "__main__":
    main()
