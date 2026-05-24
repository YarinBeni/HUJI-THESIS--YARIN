#!/usr/bin/env python3
"""Phase C: Leave-One-Ruler-Out (LORO) honesty pass.

For a given (method, cleaning, pool, layer), holds out one ruler at a time,
refits the geodesic pipeline on the remaining fragments, projects held-out
fragments via Isomap.transform(), then scores cross-ruler pairwise-order
accuracy.

Usage:
    python v_1/src/geodesic/phase_c/loro.py \
        --method thalesian_cunei400m --cleaning maximal --pool mean \
        --layer 7 --ruler Ashurbanipal

Saves results to:
    v_1/src/geodesic/results/phase_c/loro_<method>_<cleaning>_<pool>_L<layer>_<ruler>.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr as _spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "v_1/src/geodesic"))

from utils import (
    find_acts_dir,
    load_layer,
    load_year_labels,
    build_knn_graph,
    pca_l2,
    sign_flip_coord,
    pairwise_order_acc_fast,
)


def _sp(a, b) -> float:
    res = _spearmanr(a, b)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def isomap_transform(X_train: np.ndarray, X_test: np.ndarray,
                     k: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    """Fit Isomap on X_train, transform both X_train and X_test."""
    iso = Isomap(n_neighbors=k, n_components=1, metric=metric, eigen_solver="dense")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord_train = iso.fit_transform(X_train).ravel()
        coord_test  = iso.transform(X_test).ravel()
    return coord_train, coord_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",   required=True)
    ap.add_argument("--cleaning", required=True)
    ap.add_argument("--pool",     required=True)
    ap.add_argument("--layer",    type=int, required=True)
    ap.add_argument("--ruler",    required=True, help="Ruler name to hold out")
    ap.add_argument("--min-fragments", type=int, default=10,
                    help="Skip ruler if fewer held-out fragments")
    ap.add_argument("--output-dir", default="v_1/src/geodesic/results/phase_c")
    ap.add_argument("--parquet",    default="v_1/data/evaluation/corpora/orcc_corpus.parquet")
    args = ap.parse_args()

    acts_dir = find_acts_dir(args.method, args.cleaning, args.pool)
    if acts_dir is None:
        print(f"SKIP: activations not found for {args.method}/{args.cleaning}/{args.pool}")
        sys.exit(0)

    import pandas as pd
    df = pd.read_parquet(ROOT / args.parquet)
    mask_year = df["year"].notna()
    frag_idx  = np.where(mask_year)[0]
    years     = df["year"][mask_year].values.astype(float)
    rulers    = df["ruler"][mask_year].values

    # Check held-out ruler has enough fragments
    held_out_mask = rulers == args.ruler
    n_held = held_out_mask.sum()
    if n_held < args.min_fragments:
        print(f"SKIP: ruler '{args.ruler}' has only {n_held} fragments (< {args.min_fragments})")
        sys.exit(0)

    print(f"Ruler '{args.ruler}': {n_held} held-out / {len(years)-n_held} held-in fragments")

    X_raw = load_layer(acts_dir, args.layer)
    X_all = X_raw[frag_idx]

    # Baseline: full-corpus geodesic (reference for comparison)
    X_full_pca = pca_l2(X_all, n_components=64)
    k_full, _ = build_knn_graph(X_full_pca, metric="cosine")
    iso_full = Isomap(n_neighbors=k_full, n_components=1, metric="cosine", eigen_solver="dense")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord_full = iso_full.fit_transform(X_full_pca).ravel()
    coord_full = sign_flip_coord(coord_full, years)
    pacc_full  = pairwise_order_acc_fast(coord_full, years, margin=100)
    sp_full    = _sp(coord_full, years)

    # LORO: hold out this ruler
    held_in_mask  = ~held_out_mask
    X_in   = X_all[held_in_mask]
    X_out  = X_all[held_out_mask]
    y_in   = years[held_in_mask]
    y_out  = years[held_out_mask]

    # Fit PCA on held-in only, transform both
    n_comp = min(64, X_in.shape[0] - 1, X_in.shape[1])
    scaler = StandardScaler().fit(X_in)
    pca    = PCA(n_components=n_comp, random_state=42).fit(scaler.transform(X_in))

    X_in_pca  = normalize(pca.transform(scaler.transform(X_in)),  norm="l2")
    X_out_pca = normalize(pca.transform(scaler.transform(X_out)), norm="l2")

    k_loro, _ = build_knn_graph(X_in_pca, metric="cosine")

    try:
        coord_in, coord_out = isomap_transform(X_in_pca, X_out_pca, k=k_loro)
    except Exception as e:
        print(f"Isomap.transform failed: {e}")
        sys.exit(1)

    # Sign-flip using held-in labels, then apply same sign to held-out
    r_in, _ = _spearmanr(coord_in, y_in) if len(y_in) > 1 else (1.0, None)
    r_in = float(r_in.statistic if hasattr(r_in, "statistic") else r_in) if not isinstance(r_in, float) else r_in
    if r_in < 0:
        coord_in  = -coord_in
        coord_out = -coord_out

    # Pairwise-order acc on cross-ruler pairs: (held-out i, held-in j) only
    n_in  = len(y_in)
    n_out = len(y_out)
    correct = total = 0
    for oi in range(n_out):
        for ii in range(n_in):
            dy = y_out[oi] - y_in[ii]
            if abs(dy) <= 100:
                continue
            dc = coord_out[oi] - coord_in[ii]
            if np.sign(dc) == np.sign(dy):
                correct += 1
            total += 1
    pacc_loro_cross = correct / total if total > 0 else float("nan")

    # Full LORO score: concat coords, score all pairs (including within held-in)
    coord_concat = np.empty(len(years))
    coord_concat[held_in_mask]  = coord_in
    coord_concat[held_out_mask] = coord_out
    sp_loro    = _sp(coord_concat, years)
    pacc_loro  = pairwise_order_acc_fast(coord_concat, years, margin=100)

    drop_pacc  = pacc_full - pacc_loro
    drop_cross = pacc_full - pacc_loro_cross

    result = {
        "method":   args.method,
        "cleaning": args.cleaning,
        "pool":     args.pool,
        "layer":    args.layer,
        "ruler":    args.ruler,
        "n_held_out": int(n_held),
        "n_held_in":  int(held_in_mask.sum()),
        "k_full":  int(k_full),
        "k_loro":  int(k_loro),
        "pacc_full":       pacc_full,
        "sp_full":         sp_full,
        "pacc_loro":       pacc_loro,
        "sp_loro":         sp_loro,
        "pacc_loro_cross": pacc_loro_cross,
        "drop_pacc":       drop_pacc,
        "drop_cross":      drop_cross,
    }

    print(f"pacc_full={pacc_full:.4f}  pacc_loro={pacc_loro:.4f}  "
          f"pacc_cross={pacc_loro_cross:.4f}  drop={drop_pacc:.4f}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ruler_slug = args.ruler.replace(" ", "_").replace("-", "_")
    out_file = out_dir / (
        f"loro_{args.method}_{args.cleaning}_{args.pool}"
        f"_L{args.layer:02d}_{ruler_slug}.json"
    )
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
