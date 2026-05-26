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


def _loro_one_ruler(X_all, years, rulers, ruler, min_fragments):
    """Run leave-one-ruler-out for a single ruler on an already-subset matrix.

    `X_all` rows align 1:1 with `years` / `rulers`. Returns a per-ruler result
    dict, or None if the ruler has too few held-out fragments. This is the same
    computation the imbalanced single-ruler `main()` path performs, factored out
    so balanced mode can loop it over every ruler within each draw.
    """
    held_out_mask = rulers == ruler
    n_held = int(held_out_mask.sum())
    if n_held < min_fragments:
        return None

    # Baseline: full-(sub)corpus geodesic
    X_full_pca = pca_l2(X_all, n_components=64)
    k_full, _ = build_knn_graph(X_full_pca, metric="cosine")
    iso_full = Isomap(n_neighbors=k_full, n_components=1, metric="cosine", eigen_solver="dense")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord_full = iso_full.fit_transform(X_full_pca).ravel()
    coord_full = sign_flip_coord(coord_full, years)
    pacc_full  = pairwise_order_acc_fast(coord_full, years, margin=100)
    sp_full    = _sp(coord_full, years)

    held_in_mask = ~held_out_mask
    X_in  = X_all[held_in_mask]
    X_out = X_all[held_out_mask]
    y_in  = years[held_in_mask]
    y_out = years[held_out_mask]

    n_comp = min(64, X_in.shape[0] - 1, X_in.shape[1])
    scaler = StandardScaler().fit(X_in)
    pca    = PCA(n_components=n_comp, random_state=42).fit(scaler.transform(X_in))
    X_in_pca  = normalize(pca.transform(scaler.transform(X_in)),  norm="l2")
    X_out_pca = normalize(pca.transform(scaler.transform(X_out)), norm="l2")

    k_loro, _ = build_knn_graph(X_in_pca, metric="cosine")
    coord_in, coord_out = isomap_transform(X_in_pca, X_out_pca, k=k_loro)

    r_in, _ = _spearmanr(coord_in, y_in) if len(y_in) > 1 else (1.0, None)
    r_in = float(r_in.statistic if hasattr(r_in, "statistic") else r_in) if not isinstance(r_in, float) else r_in
    if r_in < 0:
        coord_in  = -coord_in
        coord_out = -coord_out

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

    coord_concat = np.empty(len(years))
    coord_concat[held_in_mask]  = coord_in
    coord_concat[held_out_mask] = coord_out
    sp_loro   = _sp(coord_concat, years)
    pacc_loro = pairwise_order_acc_fast(coord_concat, years, margin=100)

    return {
        "ruler": ruler,
        "n_held_out": n_held,
        "n_held_in":  int(held_in_mask.sum()),
        "k_full": int(k_full),
        "k_loro": int(k_loro),
        "pacc_full":       pacc_full,
        "sp_full":         sp_full,
        "pacc_loro":       pacc_loro,
        "sp_loro":         sp_loro,
        "pacc_loro_cross": pacc_loro_cross,
        "drop_pacc":       pacc_full - pacc_loro,
        "drop_cross":      pacc_full - pacc_loro_cross,
    }


# ---------------------------------------------------------------------------
# Balanced-Monte-Carlo helpers (C11)
# ---------------------------------------------------------------------------

def _parse_draw_range(s, n):
    """Inclusive 'A-B' range -> list[int]. None => all n rows."""
    if s is None:
        return list(range(n))
    a, b = s.split("-", 1)
    return list(range(int(a), int(b) + 1))


def _draw_positions(draws_matrix, draw_idx):
    """Integer positions (parquet/activation row order) for one draw.
    Mirrors `_draw_subset` in run_mc_probes.py."""
    row = draws_matrix[draw_idx]
    if row.dtype == bool:
        return np.where(row)[0]
    return row.astype(int)


def run_balanced(acts_dir, args):
    """C11 balanced mode: leave-one-ruler-out within each balanced draw.

    For each draw, restrict fragments to that draw's ~168 positions, run LORO
    over every ruler with >= --min-fragments held-out fragments, and reduce to a
    per-draw (pacc_full, mean pacc_loro, drop). Aggregate those three scalars
    across draws into mean/std for the single (method,cleaning,pool,layer)
    config. Per-ruler detail is intentionally OMITTED in balanced mode (the
    ruler membership differs per draw and balanced draws hold ~21 frags/ruler,
    so individual held-out ruler rows are not stable/comparable across draws).

    Writes loro_robustness_balanced.json (SEPARATE from imbalanced
    loro_robustness.json). Per-draw fits are deterministic (fixed random_state).
    """
    import pandas as pd

    draws_matrix = np.load(args.draws_matrix)
    if draws_matrix.ndim != 2:
        print(f"ERROR: draws_matrix must be 2D, got {draws_matrix.shape}")
        sys.exit(1)
    n_total = draws_matrix.shape[0]

    fragment_order = json.loads(Path(args.fragment_order).read_text())
    df = pd.read_parquet(ROOT / args.parquet)
    if len(df) != len(fragment_order):
        print(f"ERROR: corpus length {len(df)} != fragment_order length "
              f"{len(fragment_order)}")
        sys.exit(1)
    if len(df) != draws_matrix.shape[1]:
        print(f"ERROR: draws_matrix width {draws_matrix.shape[1]} != corpus "
              f"length {len(df)}")
        sys.exit(1)

    years_all  = df["year"].values.astype(float)
    rulers_all = df["ruler"].values

    draw_idxs = _parse_draw_range(args.draw_range, n_total)
    print(f"Balanced LORO mode: {len(draw_idxs)} draws "
          f"(range {draw_idxs[0]}..{draw_idxs[-1]} of {n_total})")

    X_raw = load_layer(acts_dir, args.layer)

    pacc_full_vals = []
    pacc_loro_mean_vals = []
    drop_vals = []

    for di in draw_idxs:
        pos    = _draw_positions(draws_matrix, di)
        X_all  = X_raw[pos]
        years  = years_all[pos]
        rulers = rulers_all[pos]

        per_ruler = []
        for ruler in np.unique(rulers):
            try:
                r = _loro_one_ruler(X_all, years, rulers, ruler, args.min_fragments)
            except Exception as e:
                print(f"  draw {di:3d} ruler '{ruler}': FAIL {type(e).__name__}: {e}")
                r = None
            if r is not None:
                per_ruler.append(r)

        if not per_ruler:
            print(f"  draw {di:3d}: no ruler met min-fragments={args.min_fragments}; skip")
            continue

        pacc_full = per_ruler[0]["pacc_full"]  # same full-(sub)corpus fit per draw
        loro_mean = float(np.mean([r["pacc_loro"] for r in per_ruler]))
        drop      = pacc_full - loro_mean
        pacc_full_vals.append(pacc_full)
        pacc_loro_mean_vals.append(loro_mean)
        drop_vals.append(drop)
        if len(drop_vals) <= 3 or len(drop_vals) % 25 == 0:
            print(f"  draw {di:3d}: pacc_full={pacc_full:.4f} "
                  f"pacc_loro_mean={loro_mean:.4f} drop={drop:.4f} "
                  f"({len(per_ruler)} rulers)", flush=True)

    def _ms(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    pf_m, pf_s = _ms(pacc_full_vals)
    pl_m, pl_s = _ms(pacc_loro_mean_vals)
    dr_m, dr_s = _ms(drop_vals)

    summary = [{
        "method":   args.method,
        "cleaning": args.cleaning,
        "pool":     args.pool,
        "layer":    args.layer,
        "regime":   "balanced",
        "n_draws":  len(drop_vals),
        "pacc_full_mean":      pf_m, "pacc_full_std":      pf_s,
        "pacc_loro_mean_mean": pl_m, "pacc_loro_mean_std": pl_s,
        "drop_mean":           dr_m, "drop_std":           dr_s,
    }]

    # Balanced summary lives next to the imbalanced loro_robustness.json
    # (the table builder reads GEO/loro_robustness*.json), not under phase_c/.
    out_dir = ROOT / "v_1/src/geodesic/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "loro_robustness_balanced.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nBalanced LORO: pacc_full={pf_m} pacc_loro_mean={pl_m} drop={dr_m} "
          f"over {len(drop_vals)} draws")
    print(f"Saved → {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",   required=True)
    ap.add_argument("--cleaning", required=True)
    ap.add_argument("--pool",     required=True)
    ap.add_argument("--layer",    type=int, required=True)
    ap.add_argument("--ruler",    default=None, help="Ruler name to hold out (imbalanced mode only)")
    ap.add_argument("--min-fragments", type=int, default=10,
                    help="Skip ruler if fewer held-out fragments")
    ap.add_argument("--output-dir", default="v_1/src/geodesic/results/phase_c")
    ap.add_argument("--parquet",    default="v_1/data/evaluation/corpora/orcc_corpus.parquet")
    # ---- C11 balanced-Monte-Carlo flags (all optional; absent => imbalanced) ----
    ap.add_argument("--draws-matrix", default=None,
                    help="Path to draws_matrix.npy (N_draws, N_corpus). "
                         "If set, run in BALANCED mode.")
    ap.add_argument("--fragment-order", default=None,
                    help="Path to corpus_fragment_order.json (parquet row order).")
    ap.add_argument("--draw-range", default=None,
                    help="Inclusive draw index range 'A-B' (default: all draws).")
    args = ap.parse_args()

    acts_dir = find_acts_dir(args.method, args.cleaning, args.pool)

    # ---- Balanced mode branch (reached BEFORE missing-acts error) ----
    if args.draws_matrix is not None:
        print(f"=== BALANCED LORO mode (C11) for {args.method}/{args.cleaning}/{args.pool} L{args.layer} ===")
        if acts_dir is None:
            print(f"SKIP: activations not found for {args.method}/{args.cleaning}/{args.pool}")
            sys.exit(0)
        run_balanced(acts_dir, args)
        return

    # ---- Imbalanced mode (unchanged) ----
    if args.ruler is None:
        print("ERROR: --ruler is required in imbalanced mode")
        sys.exit(2)
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
