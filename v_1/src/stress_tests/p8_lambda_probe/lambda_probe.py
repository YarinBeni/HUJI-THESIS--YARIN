"""P8 core — the supervision-dial spectral probe (see MATH_NOTES.md).

Solves, per training fold and per lambda in [0,1], the generalized eigenproblem

    [ (1-lam) * Xt' Mh Xt  -  lam * Xt' Lh Xt ] v  =  gamma * Xt' D Xt v

where Mh = H K_y H / ||.||_2 (centered RBF year kernel, HSIC term) and
Lh = L / ||.||_2 (kNN heat-graph Laplacian), Xt = train-PCA-projected features
(LPP-style linear out-of-sample map). lam=1 -> Laplacian eigenmaps (pure
geometry, unsupervised); lam=0 -> supervised-PCA with year target kernel
(pure dependence, no geometry).

Readouts on the TEST fold: align1 = |Spearman(z1, y)| of the leading
coordinate, and pred = Spearman of ridge-on-Z_d predictions.

Everything (PCA, graph, bandwidths, eigenvectors, ridge) is fit on train only.
Run `python lambda_probe.py --selftest` for the synthetic two-manifold check.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import linalg as sla
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import kneighbors_graph

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
EPS = 1e-8


# ------------------------------------------------------------------ pieces ---
def _heat_graph(Xt: np.ndarray, k: int):
    """Symmetrized kNN heat-kernel affinity, Laplacian L, degree D."""
    m = len(Xt)
    k = min(k, m - 1)
    G = kneighbors_graph(Xt, k, mode="distance", include_self=False)
    d2 = G.data ** 2
    sigma2 = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    G.data = np.exp(-d2 / sigma2)
    W = G.toarray()
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    return L, deg


def _year_kernel(y: np.ndarray):
    """Centered RBF kernel on the year, median-heuristic bandwidth."""
    dy = np.abs(y[:, None] - y[None, :])
    pos = dy[dy > 0]
    sig = np.median(pos) if len(pos) else 1.0
    Ky = np.exp(-(dy ** 2) / (2 * sig ** 2))
    m = len(y)
    H = np.eye(m) - np.ones((m, m)) / m
    return H @ Ky @ H


def _specnorm(A: np.ndarray) -> float:
    v = float(np.linalg.norm(A, 2))
    return v if v > EPS else 1.0


class LambdaProbe:
    """Fit once per training fold; solve() per lambda reuses the cached pieces."""

    def __init__(self, X_tr, y_tr, k=10, d=3, r=100):
        self.d = d
        r = min(r, len(X_tr) - 1, X_tr.shape[1])
        self.pca = PCA(n_components=r, random_state=0).fit(X_tr)
        Xt = self.pca.transform(X_tr)
        # scale to unit mean-norm so ridge/eig conditioning is stable
        self.scale = np.sqrt((Xt ** 2).sum(axis=1).mean()) or 1.0
        self.Xt = Xt / self.scale
        self.y = y_tr.astype(float)

        L, deg = _heat_graph(self.Xt, k)
        M = _year_kernel(self.y)
        self.Mh = M / _specnorm(M)
        self.Lh = L / _specnorm(L)
        X = self.Xt
        self.XtMX = X.T @ self.Mh @ X
        self.XtLX = X.T @ self.Lh @ X
        self.XtDX = (X * deg[:, None]).T @ X
        self.XtDX += EPS * (np.trace(self.XtDX) / len(self.XtDX)) * np.eye(len(self.XtDX))

    def solve(self, lam: float):
        """-> V (r,d): top-d generalized eigenvectors of the lambda-mixed pencil,
        trivial (near-constant embedding) directions deflated."""
        A = (1.0 - lam) * self.XtMX - lam * self.XtLX
        A = (A + A.T) / 2
        w, V = sla.eigh(A, (self.XtDX + self.XtDX.T) / 2)
        order = np.argsort(w)[::-1]
        keep = []
        for idx in order:
            z = self.Xt @ V[:, idx]
            if z.std() < 1e-10:          # near-constant = trivial direction
                continue
            keep.append(idx)
            if len(keep) == self.d:
                break
        return V[:, keep]

    def transform(self, X, V):
        return (self.pca.transform(X) / self.scale) @ V


def _sp(a, b):
    if len(a) < 3 or len(set(np.round(b, 6))) < 2 or len(set(np.round(a, 6))) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def eval_fold(X_tr, y_tr, X_te, y_te, lambdas=LAMBDAS, k=10, d=3, ridge_alpha=1.0):
    """-> {lam: {"align1":…, "pred":…}} on the test fold."""
    probe = LambdaProbe(X_tr, y_tr, k=k, d=d)
    out = {}
    for lam in lambdas:
        V = probe.solve(lam)
        Z_tr, Z_te = probe.Xt @ V, probe.transform(X_te, V)
        a1 = abs(_sp(Z_te[:, 0], y_te)) if Z_te.shape[1] else float("nan")
        reg = Ridge(alpha=ridge_alpha).fit(Z_tr, y_tr)
        pr = _sp(reg.predict(Z_te), y_te)
        out[lam] = {"align1": a1, "pred": pr}
    return out


def mc_lambda_probe(X, y, g, draw_rows_list, lambdas=LAMBDAS, k=10, d=3,
                    n_splits=5, l2_normalize=True, verbose=False):
    """Balanced-MC evaluation, mirroring shared/mc_probe.py aggregation:
    GroupKFold-by-ruler within each draw, fold-mean per draw, mean+-std over
    draws. l2_normalize: row-normalize (the suite standard for activations;
    the synthetic selftest disables it — see selftest()).
    -> {"per_lambda": {lam: {align1_mean/std, pred_mean/std}}, ...}"""
    acc = {lam: {"align1": [], "pred": []} for lam in lambdas}
    used = 0
    for di, rows in enumerate(draw_rows_list):
        Xs, ys, gs = X[rows], y[rows], g[rows]
        m = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        Xs, ys, gs = Xs[m], ys[m], gs[m]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        if l2_normalize:
            norms = np.linalg.norm(Xs, axis=1, keepdims=True)
            Xn = Xs / np.maximum(norms, EPS)
        else:
            Xn = Xs
        fold_acc = {lam: {"align1": [], "pred": []} for lam in lambdas}
        gkf = GroupKFold(n_splits=min(n_splits, nr))
        for tr, te in gkf.split(Xn, ys, gs):
            if len(set(ys[te].tolist())) < 2:
                continue
            try:
                res = eval_fold(Xn[tr], ys[tr], Xn[te], ys[te],
                                lambdas=lambdas, k=k, d=d)
            except Exception:
                continue
            for lam in lambdas:
                fold_acc[lam]["align1"].append(res[lam]["align1"])
                fold_acc[lam]["pred"].append(res[lam]["pred"])
        got = False
        for lam in lambdas:
            for key in ("align1", "pred"):
                v = [x for x in fold_acc[lam][key] if x == x]
                if v:
                    acc[lam][key].append(float(np.mean(v))); got = True
        used += got
        if verbose and (di + 1) % 25 == 0:
            print(f"    draw {di+1}/{len(draw_rows_list)}", flush=True)
    if not used:
        return {"skipped": True}

    def agg(vals):
        v = [x for x in vals if x == x]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2

    per_lambda = {}
    for lam in lambdas:
        a_m, a_s = agg(acc[lam]["align1"]); p_m, p_s = agg(acc[lam]["pred"])
        per_lambda[f"{lam:.1f}"] = {"align1_mean": a_m, "align1_std": a_s,
                                    "pred_mean": p_m, "pred_std": p_s}
    return {"n_draws_used": used, "k_neighbors": k, "d": d,
            "per_lambda": per_lambda}


# ---------------------------------------------------------------- selftest ---
def _make_manifold(n=400, y_is_dominant=True, seed=0):
    """S-curve in R^30. If y_is_dominant, year = arc-length (the manifold's big
    axis). Else, year = the SMALL transverse axis: geometry alone can't find it."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 3 * np.pi, n)          # long axis (arc length)
    t = rng.uniform(0, 1, n)                  # short transverse axis
    X3 = np.stack([np.sin(s) * (1 + 0.1 * s), 0.4 * s, t], axis=1)
    A = rng.normal(size=(3, 30)) / np.sqrt(3)
    X = X3 @ A + 0.02 * rng.normal(size=(n, 30))
    y = (s if y_is_dominant else t) + 0.01 * rng.normal(size=n)
    # random groups: the selftest validates the solver + fold plumbing, not the
    # extrapolation hardness of real ruler-groups (that's the corpus's property)
    g = rng.integers(0, 8, size=n)
    return X.astype(np.float32), y, g.astype(str)


def selftest():
    rng = np.random.default_rng(1)
    for dom, name, expect in [(True, "y = dominant manifold axis", "flat/high"),
                              (False, "y = minor transverse axis", "rising as lam->0")]:
        X, y, g = _make_manifold(y_is_dominant=dom)
        rows = [rng.choice(len(X), size=200, replace=False) for _ in range(8)]
        res = mc_lambda_probe(X, y, g, rows, k=10, d=3, n_splits=4,
                              l2_normalize=False)
        pl = res["per_lambda"]
        line = "  ".join(f"lam={lam}: a1={v['align1_mean']:.2f}/pr={v['pred_mean']:.2f}"
                         for lam, v in pl.items() if lam in ("0.0", "0.5", "1.0"))
        print(f"[{name}] expect {expect}\n  {line}")
        a0, a1 = pl["0.0"]["align1_mean"], pl["1.0"]["align1_mean"]
        if dom:
            assert a1 > 0.85, f"unsupervised end should find the dominant axis (got {a1:.2f})"
            assert a0 > 0.85, f"supervised end should also find it (got {a0:.2f})"
        else:
            assert a0 > a1 + 0.25, f"dial must matter when y is off-axis ({a0:.2f} vs {a1:.2f})"
            assert a0 > 0.7, f"supervision should dig out the transverse axis (got {a0:.2f})"
    print("selftest PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        sys.exit("this module is a library; use --selftest or run_tfidf_local.py")
