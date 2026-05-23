"""Shared geodesic utilities for Phases A–D.

All functions work on CPU; no GPU or cluster required.
Typical flow per (method, cleaning, pool, layer):
    X, y = load_layer(acts_dir, layer)
    X_pca = pca_l2(X, n_components=64)
    k, graph = build_knn_graph(X_pca, metric="cosine")
    dist = geodesic_dist(graph)
    coord_isomap = isomap_1d(X_pca, k)
    coord_ebin   = earliest_bin_coord(dist, y, bin_width=100)
    acc = pairwise_order_acc(coord_isomap, y, margin=100)
    purity, null_mean, null_std = neighbor_purity(coord_isomap, y)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Activation loading
# ---------------------------------------------------------------------------

ACTS_BASES = [
    "v_1/src/linear_probing/results/orcc__embed/activations",
    "v_1/src/linear_probing/results/orcc_round1/activations",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_acts_dir(method: str, cleaning: str, pool: str,
                  repo_root: Optional[Path] = None) -> Optional[Path]:
    root = repo_root or _repo_root()
    dir_name = f"{method}_{cleaning}_{pool}"
    for base in ACTS_BASES:
        candidate = root / base / dir_name
        if candidate.is_dir() and any(candidate.glob("layer_*.npz")):
            return candidate
    return None


def load_layer(acts_dir: Path, layer: int) -> np.ndarray:
    """Return (n_fragments, d_model) float32 array."""
    npz_path = acts_dir / f"layer_{layer:02d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    arr = np.load(npz_path)
    key = arr.files[0]
    return arr[key].astype(np.float32)


def available_layers(acts_dir: Path) -> list[int]:
    return sorted(
        int(p.stem.split("_")[1]) for p in acts_dir.glob("layer_*.npz")
    )


def load_year_labels(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (fragment_indices, years) for rows with non-null year."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    mask = df["year"].notna()
    return np.where(mask)[0], df["year"][mask].values.astype(float)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def pca_l2(X: np.ndarray, n_components: int = 64) -> np.ndarray:
    """Center → PCA → L2-normalize. Returns (n, n_components)."""
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    X_c = X - X.mean(axis=0)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_c)
    return normalize(X_pca, norm="l2")


# ---------------------------------------------------------------------------
# kNN graph + geodesic distance
# ---------------------------------------------------------------------------

def build_knn_graph(X: np.ndarray, k_min: int = 3, k_max: int = 50,
                    metric: str = "cosine") -> tuple[int, sparse.csr_matrix]:
    """Find smallest k ∈ [k_min, k_max] that makes the kNN graph connected.

    Returns (k_used, symmetric_adjacency).
    Falls back to mutual=False (union) if connectivity is never reached.
    """
    for k in range(k_min, k_max + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = kneighbors_graph(X, n_neighbors=k, metric=metric,
                                 mode="connectivity", include_self=False)
        g_sym = (g + g.T)
        g_sym.data = np.ones_like(g_sym.data)
        n_comp, _ = connected_components(g_sym, directed=False)
        if n_comp == 1:
            return k, g_sym
    # Never fully connected — return last attempt with a warning
    warnings.warn(f"kNN graph not connected at k={k_max}; using k={k_max} anyway.")
    return k_max, g_sym


def geodesic_dist(adj: sparse.csr_matrix) -> np.ndarray:
    """All-pairs geodesic distance matrix via BFS/Dijkstra."""
    return shortest_path(adj, method="D", directed=False, unweighted=True)


# ---------------------------------------------------------------------------
# 1D coordinates
# ---------------------------------------------------------------------------

def isomap_1d(X: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Isomap 1D embedding. Returns (n,) coordinate, sign-flipped if needed."""
    iso = Isomap(n_neighbors=k, n_components=1, metric=metric,
                 eigen_solver="dense")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord = iso.fit_transform(X).ravel()
    return coord


def earliest_bin_coord(dist_matrix: np.ndarray, years: np.ndarray,
                       bin_width: int = 100) -> np.ndarray:
    """Average geodesic distance from each fragment to all fragments in the
    earliest year bin.  Negated so that 'closer to earliest' = larger value
    before sign-flip (see sign_flip_coord).
    """
    y_min = np.nanmin(years)
    bin_mask = (years >= y_min) & (years < y_min + bin_width)
    if bin_mask.sum() == 0:
        bin_mask[np.argmin(years)] = True  # degenerate: use single oldest
    ref_dists = dist_matrix[:, bin_mask].mean(axis=1)
    # Invert: close to earliest bin → high coordinate (chronologically early)
    return -ref_dists


def sign_flip_coord(coord: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Flip sign so that Spearman(coord, years) > 0 (later = higher coord)."""
    r, _ = spearmanr(coord, years)
    return coord if r >= 0 else -coord


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def pairwise_order_acc(coord: np.ndarray, years: np.ndarray,
                       margin: int = 100) -> float:
    """Fraction of pairs (i,j) with |year_i - year_j| > margin where
    sign(coord_i - coord_j) == sign(year_i - year_j).
    """
    coord = sign_flip_coord(coord, years)
    n = len(years)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dy = years[i] - years[j]
            if abs(dy) <= margin:
                continue
            dc = coord[i] - coord[j]
            if np.sign(dc) == np.sign(dy):
                correct += 1
            total += 1
    return correct / total if total > 0 else float("nan")


def pairwise_order_acc_fast(coord: np.ndarray, years: np.ndarray,
                             margin: int = 100) -> float:
    """Vectorized version of pairwise_order_acc."""
    coord = sign_flip_coord(coord, years)
    i_idx, j_idx = np.triu_indices(len(years), k=1)
    dy = years[i_idx] - years[j_idx]
    dc = coord[i_idx] - coord[j_idx]
    valid = np.abs(dy) > margin
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(dc[valid]) == np.sign(dy[valid])))


def neighbor_purity(coord: np.ndarray, years: np.ndarray,
                    k: int = 10, window: int = 100,
                    n_perm: int = 500, rng_seed: int = 42) -> tuple[float, float, float]:
    """Temporal neighbor purity: fraction of k-nearest neighbors (in coord)
    within ±window years.

    Returns (observed_purity, null_mean, null_std).
    """
    coord = sign_flip_coord(coord, years)
    order = np.argsort(coord)
    n = len(order)
    purities = []
    for rank, idx in enumerate(order):
        lo = max(0, rank - k)
        hi = min(n, rank + k + 1)
        nbrs = [order[r] for r in range(lo, hi) if order[r] != idx]
        nbrs = sorted(nbrs, key=lambda j: abs(coord[j] - coord[idx]))[:k]
        if not nbrs:
            continue
        purities.append(np.mean(np.abs(years[nbrs] - years[idx]) <= window))
    observed = float(np.mean(purities))

    rng = np.random.default_rng(rng_seed)
    null_purities = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(years)
        order_shuf = np.argsort(coord)
        ps = []
        for rank, idx in enumerate(order_shuf):
            lo = max(0, rank - k)
            hi = min(n, rank + k + 1)
            nbrs = [order_shuf[r] for r in range(lo, hi) if order_shuf[r] != idx]
            nbrs = sorted(nbrs, key=lambda j: abs(coord[j] - coord[idx]))[:k]
            if not nbrs:
                continue
            ps.append(np.mean(np.abs(y_shuf[nbrs] - y_shuf[idx]) <= window))
        null_purities.append(np.mean(ps))
    return observed, float(np.mean(null_purities)), float(np.std(null_purities))


def pls_pairwise_acc(X: np.ndarray, y: np.ndarray, margin: int = 100) -> float:
    """Fit 1-component PLS on (X, y) and compute pairwise-order acc on full data."""
    pls = PLSRegression(n_components=1)
    pls.fit(X, y)
    pred = pls.predict(X).ravel()
    return pairwise_order_acc_fast(pred, y, margin=margin)
