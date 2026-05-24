#!/usr/bin/env python3
"""Phase D: Goodfire-style centroid + spline visualization.

Bins texts into 100-yr windows, computes PCA-3D centroids, fits a cubic
spline, and saves four publication-quality PNGs (colored by year / ruler /
archive / geodesic-coord).

Usage:
    python v_1/src/geodesic/phase_d/centroid_spline.py \
        --method thalesian_cunei400m --cleaning maximal --pool mean --layer 7
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize, StandardScaler
from scipy.stats import spearmanr as _spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "v_1/src/geodesic"))

from utils import (
    find_acts_dir,
    load_layer,
    pca_l2,
    build_knn_graph,
    isomap_1d,
    sign_flip_coord,
    pairwise_order_acc_fast,
)

BIN_WIDTH   = 100
MIN_BIN_N   = 5
N_SPLINE_PTS = 300
PCA3_COMPONENTS = 3


def _sp(a, b) -> float:
    res = _spearmanr(a, b)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def fit_spline(bin_years, bin_coords_3d, bin_counts):
    """Fit one UnivariateSpline per PCA dimension, weighted by sqrt(count)."""
    weights = np.sqrt(bin_counts)
    n = len(bin_years)
    splines = []
    for dim in range(PCA3_COMPONENTS):
        s = n * 0.001
        try:
            sp = UnivariateSpline(bin_years, bin_coords_3d[:, dim], w=weights, s=s, k=3)
        except Exception:
            sp = UnivariateSpline(bin_years, bin_coords_3d[:, dim], w=weights, s=n * 0.01, k=3)
        splines.append(sp)
    return splines


def plot_3d(ax, x, y, z, c, cmap, norm, title, label=None):
    sc = ax.scatter(x, y, z, c=c, cmap=cmap, norm=norm, s=8, alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7); ax.set_zlabel("PC3", fontsize=7)
    ax.tick_params(labelsize=6)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",   required=True)
    ap.add_argument("--cleaning", required=True)
    ap.add_argument("--pool",     required=True)
    ap.add_argument("--layer",    type=int, required=True)
    ap.add_argument("--bin-width",  type=int,   default=BIN_WIDTH)
    ap.add_argument("--min-bin-n",  type=int,   default=MIN_BIN_N)
    ap.add_argument("--output-dir", default="v_1/src/geodesic/results/phase_d")
    ap.add_argument("--parquet",    default="v_1/data/evaluation/corpora/orcc_corpus.parquet")
    args = ap.parse_args()

    acts_dir = find_acts_dir(args.method, args.cleaning, args.pool)
    if acts_dir is None:
        print(f"SKIP: activations not found for {args.method}/{args.cleaning}/{args.pool}")
        sys.exit(0)

    import pandas as pd
    df      = pd.read_parquet(ROOT / args.parquet)
    mask    = df["year"].notna()
    frag_idx = np.where(mask)[0]
    years   = df["year"][mask].values.astype(float)
    rulers  = df["ruler"][mask].values
    archives = df["archive"].values[mask] if "archive" in df.columns else np.array(["unknown"] * mask.sum())

    print(f"Loading L{args.layer} for {args.method}/{args.cleaning}/{args.pool}...")
    X_raw = load_layer(acts_dir, args.layer)
    X     = X_raw[frag_idx]

    # Geodesic pipeline for 1D coordinate
    print("PCA + kNN + Isomap...")
    X_pca64 = pca_l2(X, n_components=64)
    k, _    = build_knn_graph(X_pca64, metric="cosine")
    coord   = isomap_1d(X_pca64, k, metric="cosine")
    coord   = sign_flip_coord(coord, years)
    sp_geo  = _sp(coord, years)
    pacc    = pairwise_order_acc_fast(coord, years, margin=100)
    print(f"Geodesic: Spearman={sp_geo:.4f}  pairwise-acc={pacc:.4f}")

    # PCA-3D for visualization (fitted fresh on all fragments)
    scaler  = StandardScaler()
    X_z     = scaler.fit_transform(X)
    pca3    = PCA(n_components=PCA3_COMPONENTS, random_state=42)
    X_3d    = pca3.fit_transform(X_z)

    # Bin fragments
    year_min = np.floor(years.min() / args.bin_width) * args.bin_width
    year_max = np.ceil(years.max()  / args.bin_width) * args.bin_width
    bin_edges = np.arange(year_min, year_max + args.bin_width, args.bin_width)

    bin_centers, bin_cents_3d, bin_counts = [], [], []
    for lo in bin_edges[:-1]:
        hi   = lo + args.bin_width
        mask_b = (years >= lo) & (years < hi)
        if mask_b.sum() < args.min_bin_n:
            continue
        bc = (lo + hi) / 2
        bin_centers.append(bc)
        bin_cents_3d.append(X_3d[mask_b].mean(axis=0))
        bin_counts.append(mask_b.sum())

    bin_centers  = np.array(bin_centers)
    bin_cents_3d = np.array(bin_cents_3d)
    bin_counts   = np.array(bin_counts)
    print(f"Bins: {len(bin_centers)} bins with ≥{args.min_bin_n} fragments")

    # Spline
    splines   = fit_spline(bin_centers, bin_cents_3d, bin_counts)
    t_fine    = np.linspace(bin_centers.min(), bin_centers.max(), N_SPLINE_PTS)
    spline_3d = np.column_stack([sp(t_fine) for sp in splines])

    # Arc-length Spearman
    diffs    = np.diff(spline_3d, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc_len  = np.concatenate([[0], np.cumsum(seg_lens)])
    # Map bin_centers to arc lengths via nearest spline point
    bin_arc  = np.array([arc_len[np.argmin(np.abs(t_fine - bc))] for bc in bin_centers])
    sp_arc   = _sp(bin_arc, bin_centers)
    print(f"Arc-length Spearman vs bin year: {sp_arc:.4f}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag     = f"{args.method}_{args.cleaning}_{args.pool}_L{args.layer:02d}"

    def make_fig(color_vals, cmap, norm, color_label, plot_tag):
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")

        plot_3d(ax, X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                color_vals, cmap, norm, f"{args.method} L{args.layer} ({args.cleaning}/{args.pool})")

        # Bin centroids
        ax.scatter(bin_cents_3d[:, 0], bin_cents_3d[:, 1], bin_cents_3d[:, 2],
                   c="black", s=40, marker="D", zorder=5, label="bin centroids")

        # Spline curve
        ax.plot(spline_3d[:, 0], spline_3d[:, 1], spline_3d[:, 2],
                color="red", lw=1.5, alpha=0.8, label="spline")

        ax.legend(fontsize=7)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(color_label, fontsize=8)

        fig.text(0.5, 0.01,
                 f"Geodesic Spearman={sp_geo:.3f}  pacc={pacc:.3f}  arc-len Sp={sp_arc:.3f}",
                 ha="center", fontsize=8)

        path = out_dir / f"phase_d_{tag}_{plot_tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    # D1: colored by year
    make_fig(years, "viridis", Normalize(years.min(), years.max()), "Year", "year")

    # D2: colored by ruler
    unique_rulers = sorted(set(rulers))
    ruler_idx     = np.array([unique_rulers.index(r) for r in rulers])
    make_fig(ruler_idx, "tab20",
             Normalize(0, len(unique_rulers) - 1), "Ruler", "ruler")

    # D3: colored by archive
    unique_arch   = sorted(set(archives))
    arch_idx      = np.array([unique_arch.index(a) for a in archives])
    make_fig(arch_idx, "tab10",
             Normalize(0, len(unique_arch) - 1), "Archive", "archive")

    # D4: colored by geodesic coordinate
    make_fig(coord, "plasma", Normalize(coord.min(), coord.max()), "Geodesic coord", "geodesic")

    # Save metrics JSON
    metrics = {
        "method":   args.method, "cleaning": args.cleaning,
        "pool":     args.pool,   "layer":    args.layer,
        "geodesic_spearman":   sp_geo,
        "pairwise_order_acc":  pacc,
        "arc_length_spearman": sp_arc,
        "n_bins": len(bin_centers),
        "bin_centers": bin_centers.tolist(),
        "bin_counts":  bin_counts.tolist(),
    }
    (out_dir / f"phase_d_{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Metrics → phase_d_{tag}_metrics.json")
    print("Done.")


if __name__ == "__main__":
    main()
