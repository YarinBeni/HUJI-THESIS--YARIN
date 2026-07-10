"""P9 core — Geodesic Kernel PLS (G-KPLS), working note §2.

Isomap = kernel PCA on the doubly-centered geodesic Gram matrix
K_G = -1/2 H D_G H (Ham et al. 2004), so the one-stage supervised repair of
"Isomap then PLS" is kernel PLS (Rosipal-Trejo 2001) on K_G itself. Out-of-
sample: a test point is connected to its k nearest TRAINING points and its
geodesics run through the fixed training graph (never rebuilt -> no leakage);
its centered kernel column follows the Nystrom/Bengio formula (note eq. 5).

Three estimator arms per fold (the note's mandatory isolating baselines):
  gkpls    kernel PLS on K_G, a in {1,2,3,5} components (best-a surfaced)
  rbfkpls  kernel PLS on the Euclidean RBF kernel  (isolates geodesic vs kernel)
  krr_geo  kernel ridge on K_G, lam in {1e-3,1e-2,1e-1}  (isolates PLS vs kernel)

Run `python gkpls.py --selftest` (spiral manifold: Euclidean shortcuts across
turns, geodesic must not).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import kneighbors_graph

PLS_AS = [1, 2, 3, 5]
KRR_LAMS = [1e-3, 1e-2, 1e-1]
EPS = 1e-10


# ------------------------------------------------------------- geodesics ---
def train_geodesics(X_tr: np.ndarray, k: int):
    """Symmetrized kNN distance graph on train + all-pairs geodesics.
    Disconnected pairs fall back to (max finite geodesic) + Euclidean."""
    m = len(X_tr)
    G = kneighbors_graph(X_tr, min(k, m - 1), mode="distance",
                         include_self=False)
    G = G.maximum(G.T)
    D = shortest_path(G, method="D", directed=False)
    bad = ~np.isfinite(D)
    if bad.any():
        dmax = D[np.isfinite(D)].max()
        E = cdist(X_tr, X_tr)
        D[bad] = dmax + E[bad]
    return G, D


def test_geodesics(X_te, X_tr, G_tr, D_tr, k: int):
    """Geodesic distances test->train through the FIXED training graph:
    d_G(x*, x_i) = min_a [ d(x*, a) + D_tr(a, i) ] over the k nearest
    training anchors a of x*. One vectorized pass; graph never rebuilt."""
    E = cdist(X_te, X_tr)
    kk = min(k, len(X_tr))
    out = np.empty_like(E)
    for t in range(len(X_te)):
        anchors = np.argpartition(E[t], kk - 1)[:kk]
        out[t] = (E[t, anchors][:, None] + D_tr[anchors, :]).min(axis=0)
        out[t, anchors] = np.minimum(out[t, anchors], E[t, anchors])
    return out


def center_train_gram(D2: np.ndarray):
    """K_G = -1/2 H D2 H, eigen-clipped to PSD."""
    m = len(D2)
    H = np.eye(m) - np.ones((m, m)) / m
    K = -0.5 * H @ D2 @ H
    K = (K + K.T) / 2
    w, U = np.linalg.eigh(K)
    K = (U * np.clip(w, 0, None)) @ U.T
    return K


def center_test_cols(D2_te: np.ndarray, D2_tr: np.ndarray):
    """Note eq. (5): centered kernel columns for test points."""
    col_mean = D2_tr.mean(axis=0)          # (m,)  mean_j d2(xj, xi)
    grand = D2_tr.mean()                   # mean_{jl}
    row_mean = D2_te.mean(axis=1, keepdims=True)   # mean_j d2(x*, xj)
    return -0.5 * (D2_te - row_mean - col_mean[None, :] + grand)


def rbf_kernels(X_tr, X_te):
    """Median-heuristic RBF train Gram (double-centered) + centered test cols."""
    D2 = cdist(X_tr, X_tr, "sqeuclidean")
    pos = D2[D2 > 0]
    s2 = np.median(pos) if len(pos) else 1.0
    K = np.exp(-D2 / s2)
    Kt = np.exp(-cdist(X_te, X_tr, "sqeuclidean") / s2)
    m = len(K)
    cm = K.mean(axis=0); g = K.mean()
    Kc = K - cm[None, :] - cm[:, None] + g
    Ktc = Kt - Kt.mean(axis=1, keepdims=True) - cm[None, :] + g
    return (Kc + Kc.T) / 2, Ktc


# ------------------------------------------------------------------ KPLS ---
def kpls_fit(K: np.ndarray, y: np.ndarray, a: int):
    """Rosipal-Trejo kernel PLS, univariate y. K = centered train Gram.
    -> dual coefficients B with prediction  y_hat = k*_centered @ B + mean(y)."""
    ym = y.mean()
    yd = (y - ym).reshape(-1, 1)
    Kd = K.copy()
    T, U = [], []
    for _ in range(a):
        ny = np.linalg.norm(yd)
        if ny < EPS:
            break
        u = yd / ny
        t = Kd @ u
        nt = np.linalg.norm(t)
        if nt < EPS:
            break
        t = t / nt
        T.append(t); U.append(u)
        P = np.eye(len(K)) - t @ t.T
        Kd = P @ Kd @ P
        yd = yd - t @ (t.T @ yd)
    if not T:
        return None, ym
    T = np.hstack(T); U = np.hstack(U)
    M = T.T @ K @ U
    B = U @ np.linalg.pinv(M) @ (T.T @ (y - ym).reshape(-1, 1))
    return B, ym


def kpls_predict(Kt_c: np.ndarray, B, ym: float):
    return (Kt_c @ B).ravel() + ym


def krr_fit_predict(K, Kt_c, y, lam: float):
    """Kernel ridge via TRUNCATED spectral inverse: the PSD-clipped MDS Gram
    has a large numerical null space, and test columns are not confined to
    it — a dense solve amplifies those directions catastrophically. Invert
    on the significant eigenspace only."""
    ym = y.mean()
    w, U = np.linalg.eigh((K + K.T) / 2)
    keep = w > 1e-8 * max(w[-1], EPS)
    w, U = w[keep], U[:, keep]
    scale = w.mean() or 1.0
    alpha = U @ ((U.T @ (y - ym)) / (w + lam * scale))
    return Kt_c @ alpha + ym


# ------------------------------------------------------------------ folds ---
def _sp(a, b):
    if len(a) < 3 or len(set(np.round(b, 6))) < 2 or len(set(np.round(a, 6))) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def eval_fold(X_tr, y_tr, X_te, y_te, k=10):
    """-> {"gkpls": {a: sp}, "rbfkpls": {a: sp}, "krr_geo": {lam: sp}}"""
    G_tr, D_tr = train_geodesics(X_tr, k)
    K = center_train_gram(D_tr ** 2)
    D_te = test_geodesics(X_te, X_tr, G_tr, D_tr, k)
    Kt = center_test_cols(D_te ** 2, D_tr ** 2)
    Krbf, Krbf_t = rbf_kernels(X_tr, X_te)
    out = {"gkpls": {}, "rbfkpls": {}, "krr_geo": {}}
    for a in PLS_AS:
        if a >= len(X_tr):
            continue
        B, ym = kpls_fit(K, y_tr, a)
        out["gkpls"][a] = _sp(kpls_predict(Kt, B, ym), y_te) if B is not None else float("nan")
        B2, ym2 = kpls_fit(Krbf, y_tr, a)
        out["rbfkpls"][a] = _sp(kpls_predict(Krbf_t, B2, ym2), y_te) if B2 is not None else float("nan")
    for lam in KRR_LAMS:
        out["krr_geo"][lam] = _sp(krr_fit_predict(K, Kt, y_tr, lam), y_te)
    return out


def mc_gkpls_probe(X, y, g, draw_rows_list, k=10, n_splits=5,
                   l2_normalize=True, verbose=False):
    """Balanced-MC evaluation (same aggregation as shared/mc_probe.py):
    GroupKFold-by-ruler within each draw, fold-mean per draw, mean+-std over
    draws, best hyper surfaced per arm."""
    arms = {"gkpls": PLS_AS, "rbfkpls": PLS_AS, "krr_geo": KRR_LAMS}
    acc = {arm: {h: [] for h in hs} for arm, hs in arms.items()}
    used = 0
    for di, rows in enumerate(draw_rows_list):
        Xs, ys, gs = X[rows], y[rows], g[rows]
        mfin = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        Xs, ys, gs = Xs[mfin], ys[mfin], gs[mfin]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        if l2_normalize:
            Xs = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), EPS)
        fold = {arm: {h: [] for h in hs} for arm, hs in arms.items()}
        for tr, te in GroupKFold(n_splits=min(n_splits, nr)).split(Xs, ys, gs):
            if len(set(ys[te].tolist())) < 2:
                continue
            try:
                res = eval_fold(Xs[tr], ys[tr], Xs[te], ys[te], k=k)
            except Exception:
                continue
            for arm in arms:
                for h, v in res[arm].items():
                    fold[arm][h].append(v)
        got = False
        for arm, hs in arms.items():
            for h in hs:
                v = [x for x in fold[arm][h] if x == x]
                if v:
                    acc[arm][h].append(float(np.mean(v))); got = True
        used += got
        if verbose and (di + 1) % 25 == 0:
            print(f"    draw {di+1}/{len(draw_rows_list)}", flush=True)
    if not used:
        return {"skipped": True}

    def agg(vals):
        v = [x for x in vals if x == x]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2

    out = {"n_draws_used": used, "k_neighbors": k}
    for arm, hs in arms.items():
        per_h = {}
        for h in hs:
            mu, sd = agg(acc[arm][h])
            per_h[str(h)] = {"spearman_mean": mu, "spearman_std": sd}
        best = max(per_h, key=lambda hh: per_h[hh]["spearman_mean"]
                   if per_h[hh]["spearman_mean"] == per_h[hh]["spearman_mean"] else -9)
        out[arm] = {"per_hyper": per_h, "best_hyper": best,
                    "spearman_mean": per_h[best]["spearman_mean"],
                    "spearman_std": per_h[best]["spearman_std"]}
    return out


# ---------------------------------------------------------------- selftest ---
def selftest():
    rng = np.random.default_rng(0)
    n = 400
    th = rng.uniform(0, 4 * np.pi, n)
    r = 1 + 0.3 * th                      # wide turn gap: no kNN shortcuts
    X3 = np.stack([r * np.cos(th), r * np.sin(th),
                   0.1 * rng.uniform(0, 1, n)], axis=1)
    Q, _ = np.linalg.qr(rng.normal(size=(25, 3)))   # isometric 3->25 lift
    X = (X3 @ Q.T + 0.01 * rng.normal(size=(n, 25))).astype(np.float32)
    y = th                                  # arc length = the label
    g = rng.integers(0, 8, size=n).astype(str)
    rows = [rng.choice(n, size=200, replace=False) for _ in range(8)]
    res = mc_gkpls_probe(X, y, g, rows, k=8, n_splits=4, l2_normalize=False)
    gk, rb, kr = res["gkpls"], res["rbfkpls"], res["krr_geo"]
    print(f"spiral: gkpls={gk['spearman_mean']:.3f}(a={gk['best_hyper']})"
          f"  rbfkpls={rb['spearman_mean']:.3f}"
          f"  krr_geo={kr['spearman_mean']:.3f}(lam={kr['best_hyper']})")
    assert gk["spearman_mean"] > 0.9, "G-KPLS must recover arc length on the spiral"
    assert kr["spearman_mean"] > 0.85, "KRR on K_G should also recover it"
    print("selftest PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        sys.exit("library module; use --selftest or run_acts.py")
