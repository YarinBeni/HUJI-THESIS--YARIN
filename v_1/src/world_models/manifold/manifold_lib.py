"""Representation-manifold diagnostics (Modell et al. 2025, arXiv 2505.18235).

The claim being tested: a continuous feature (year; lon/lat) is not one direction but a
low-dimensional MANIFOLD, and the representation map is approximately an isometry from
the feature's intrinsic metric onto that manifold. Two empirical signatures:

  (1) local   cos(x_i, x_j) ~ 1 - 0.5 * d_feature(i,j)^2      -> Chatterjee xi
  (2) global  graph-geodesic distance ~ d_feature(i,j)         -> Pearson rho

We follow their pipeline: drop degenerate rows -> L2 row-normalise -> (optionally) project
onto the top-r UNCENTERED singular directions and renormalise -> kNN graph with Euclidean
edge weights -> Dijkstra for the manifold metric. PCA is used only for the 3-D pictures.

Feature metrics here (ours, not theirs):
  * year  -> |y_i - y_j|, plus monotone reparameterisations (log-recency), since their
             years case needed log(2019 - year) before the isometry appeared.
  * geo   -> great-circle (haversine) distance, the intrinsic metric of the sphere; the
             analogue of their circular hue/day-of-year metrics.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

EPS = 1e-3


# --------------------------------------------------------------------------- prep
def prep(X, r=None, eps=EPS):
    """Drop near-zero rows, L2-normalise; optionally project onto the top-r uncentered
    singular directions and renormalise (their denoising step for high-d embeddings).

    Returns (Xn, keep_mask)."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    keep = norms > eps
    Xn = X[keep] / norms[keep][:, None]
    if r:
        r = min(r, min(Xn.shape) - 1)
        u, s, _ = svds(Xn, k=r)
        Xl = (u * s)[:, ::-1]                       # svds returns ascending
        n2 = np.linalg.norm(Xl, axis=1)
        n2[n2 == 0] = 1.0
        Xn = Xl / n2[:, None]
    return Xn, keep


def knn_graph(X, k, clamp=1e-3):
    """Symmetric kNN graph, Euclidean edge weights (their `knn_graph`)."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    d, i = nn.kneighbors(X)
    n = len(X)
    rows = np.repeat(np.arange(n), d.shape[1])
    A = csr_matrix((d.ravel(), (rows, i.ravel())), shape=(n, n))
    A = A.maximum(A.T)
    if clamp:
        A.data[A.data < clamp] = clamp
    A.eliminate_zeros()
    return A


def lcc_mask(A):
    """Boolean mask of the largest connected component + (n_components, size)."""
    ncomp, lab = connected_components(A, directed=False)
    if ncomp == 1:
        return np.ones(A.shape[0], dtype=bool), ncomp, A.shape[0]
    sizes = np.bincount(lab)
    big = int(sizes.argmax())
    return lab == big, ncomp, int(sizes[big])


def manifold_distance(X, k):
    """Graph-geodesic (Dijkstra) distances; restricted to the largest component.
    Returns (D_manifold, keep_mask_within_X, n_components)."""
    A = knn_graph(X, k)
    m, ncomp, _ = lcc_mask(A)
    if not m.all():
        A = knn_graph(X[m], k)
    D = dijkstra(A, directed=False)
    return D, m, ncomp


# ------------------------------------------------------------------- feature metrics
def year_metric(y, kind="abs", ref=None):
    """Pairwise feature distance for a scalar year target.

    kind: 'abs'      -> |y_i - y_j|                       (raw)
          'log'      -> |log(ref - y_i) - log(ref - y_j)| (their log-recency analogue;
                        ref defaults to max(y)+1 so all arguments stay positive)
          'sqrt'     -> |sqrt(ref-y_i) - sqrt(ref-y_j)|
    """
    y = np.asarray(y, dtype=float)
    if kind == "abs":
        t = y
    else:
        ref = float(np.nanmax(y) + 1) if ref is None else float(ref)
        z = np.clip(ref - y, 1e-9, None)
        t = np.log(z) if kind == "log" else np.sqrt(z)
    return squareform(pdist(t[:, None]))


def haversine_metric(lonlat, radius=6371.0):
    """Great-circle distance (km) between (lon, lat) rows — the sphere's intrinsic
    metric, our analogue of their circular hue/date metrics."""
    a = np.asarray(lonlat, dtype=float)
    lon = np.radians(a[:, 0]); lat = np.radians(a[:, 1])
    sla, cla = np.sin(lat), np.cos(lat)
    # cos(central angle) = sin.sin + cos.cos.cos(dlon)
    c = (sla[:, None] * sla[None, :]
         + cla[:, None] * cla[None, :] * np.cos(lon[:, None] - lon[None, :]))
    return radius * np.arccos(np.clip(c, -1.0, 1.0))


def cosine_similarity_matrix(X):
    return 1.0 - squareform(pdist(X, metric="cosine"))


# ------------------------------------------------------------------------ statistics
def chatterjee_corr(x, y):
    """Chatterjee's xi — rank-based dependence, invariant to monotone maps of x."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    idx = np.argsort(x, kind="mergesort")
    rank_y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    r = rank_y[idx]
    S = np.sum(np.abs(np.diff(r)))
    return float(1.0 - (3.0 * S) / (n ** 2 - 1))


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def pair_summary(Dfeat, Drep, n_bins=60, square_x=False):
    """Their `distance_plot` reduction, generalised to continuous features.

    They bin by the unique feature-distance value and average the representation
    distance within each bin; with real-valued years / km we bin into `n_bins` quantile
    bins instead. Returns (x_binned, y_mean, x_pairs, y_pairs) — the first two for
    plotting, the last two (raw upper-triangle pairs) for the correlation.
    """
    iu = np.triu_indices_from(Dfeat, k=1)
    x = Dfeat[iu]; y = Drep[iu]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) == 0:
        return np.array([]), np.array([]), x, y
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return x, y, x, y
    ib = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    xb = np.array([x[ib == b].mean() for b in range(len(qs) - 1) if (ib == b).any()])
    yb = np.array([y[ib == b].mean() for b in range(len(qs) - 1) if (ib == b).any()])
    if square_x:
        xb = xb ** 2
    return xb, yb, x, y


def isometry_stats(X, feat_D, k=10):
    """The two headline numbers for one (representation, feature-metric) pair.

    xi   : Chatterjee between SQUARED feature distance and cosine similarity  (local)
    rho  : Pearson  between feature distance and graph-geodesic distance      (global)
    """
    Dm, keep, ncomp = manifold_distance(X, k)
    fD = feat_D[np.ix_(keep, keep)]
    Xc = X[keep]
    cos = cosine_similarity_matrix(Xc)
    iu = np.triu_indices_from(fD, k=1)
    xi = chatterjee_corr(fD[iu], cos[iu])          # xi is monotone-invariant in x
    fin = np.isfinite(Dm[iu])
    rho = pearson(fD[iu][fin], Dm[iu][fin])
    return {"xi_cos": xi, "rho_geodesic": rho, "n": int(keep.sum()),
            "n_components": int(ncomp), "k": int(k)}
