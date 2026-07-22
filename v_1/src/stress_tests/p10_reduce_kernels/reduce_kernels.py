"""P10 core — the advisor's "reduce, then kernel/dial" idea.

Hypothesis: the chronology manifold may be easier to read after a low-dim
embedding. So, *inside each CV fold* (fit on train, apply to test — no leakage), we
    1. reduce the activations to `dims` (default 3) with one of raw / PCA / PLS / UMAP
       (PLS is supervised → uses y_tr only; PCA/UMAP unsupervised),
    2. optionally normalize the reduced coords (none / zscore / l2, fit on train),
then run the SAME estimators as P9/P8 on the reduced features:
    * gkpls / rbfkpls / krr_geo  (reused from p9_gkpls.gkpls.eval_fold)
    * the supervision dial        (reused from p8_lambda_probe.lambda_probe.eval_fold)

Balanced-MC aggregation identical to P8/P9: GroupKFold-by-ruler within each of the
200 balanced draws, fold-mean per draw, mean±std over draws. `raw` reproduces the
existing P9/P8 (sanity anchor). t-SNE is viz-only (no train→test transform) and lives
in plot_reductions.py, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "p9_gkpls"))
sys.path.insert(0, str(_HERE.parents[1] / "p8_lambda_probe"))
from gkpls import PLS_AS, KRR_LAMS, eval_fold as gk_eval          # noqa: E402
from lambda_probe import eval_fold as dial_eval                    # noqa: E402

EPS = 1e-8
REDUCERS = ["raw", "pca", "pls", "umap"]
NORMS = ["none", "zscore", "l2"]
DIAL_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # coarser grid for speed


def _fit_reducer(kind, dims, X_tr, y_tr, seed=42):
    d = min(dims, X_tr.shape[1] - 1, max(2, len(X_tr) - 1))
    if kind == "raw":
        return None
    if kind == "pca":
        return PCA(n_components=d, random_state=seed).fit(X_tr)
    if kind == "pls":
        return PLSRegression(n_components=d, scale=False).fit(X_tr, y_tr)
    if kind == "umap":
        import umap  # noqa
        return umap.UMAP(n_components=dims, n_neighbors=15, min_dist=0.1,
                         random_state=seed).fit(X_tr)
    raise ValueError(kind)


def _apply(kind, r, X):
    return X if r is None else np.asarray(r.transform(X), dtype=float)


def _fit_norm(kind, Z_tr):
    if kind == "zscore":
        mu, sd = Z_tr.mean(0), Z_tr.std(0)
        return mu, np.where(sd == 0, 1.0, sd)
    return None


def _apply_norm(kind, p, Z):
    if kind == "zscore":
        mu, sd = p
        return (Z - mu) / sd
    if kind == "l2":
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), EPS)
    return Z


def _reduce_fold(reducer, norm, dims, X_tr, y_tr, X_te):
    r = _fit_reducer(reducer, dims, X_tr, y_tr)
    Ztr, Zte = _apply(reducer, r, X_tr), _apply(reducer, r, X_te)
    p = _fit_norm(norm, Ztr)
    return _apply_norm(norm, p, Ztr), _apply_norm(norm, p, Zte)


def _agg(vals):
    v = [x for x in vals if x == x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def mc_reduced(X, y, g, draw_rows, reducer="pca", norm="none", dims=3, k=10,
               n_splits=5, do_gkpls=True, do_dial=True, dial_lambdas=DIAL_LAMBDAS):
    """Reduce-then-{gkpls,rbfkpls,krr,dial} under balanced-MC. Returns a dict with a
    `gkpls`/`rbfkpls`/`krr_geo` block (best hyper, spearman mean±std) and a `dial`
    block (per-lambda align1 & pred mean±std)."""
    gk_arms = {"gkpls": PLS_AS, "rbfkpls": PLS_AS, "krr_geo": KRR_LAMS}
    acc_gk = {a: {h: [] for h in hs} for a, hs in gk_arms.items()}
    acc_dl = {lam: {"align1": [], "pred": []} for lam in dial_lambdas}
    used = 0
    for rows in draw_rows:
        Xs, ys, gs = X[rows], y[rows], g[rows]
        m = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        Xs, ys, gs = Xs[m], ys[m], gs[m]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        fgk = {a: {h: [] for h in hs} for a, hs in gk_arms.items()}
        fdl = {lam: {"align1": [], "pred": []} for lam in dial_lambdas}
        for tr, te in GroupKFold(n_splits=min(n_splits, nr)).split(Xs, ys, gs):
            if len(set(ys[te].tolist())) < 2:
                continue
            try:
                Ztr, Zte = _reduce_fold(reducer, norm, dims, Xs[tr], ys[tr], Xs[te])
            except Exception:
                continue
            if do_gkpls:
                try:
                    rg = gk_eval(Ztr, ys[tr], Zte, ys[te], k=k)
                    for a in gk_arms:
                        for h, v in rg[a].items():
                            fgk[a][h].append(v)
                except Exception:
                    pass
            if do_dial:
                try:
                    rd = dial_eval(Ztr, ys[tr], Zte, ys[te],
                                   lambdas=dial_lambdas, k=k,
                                   d=min(dims, Ztr.shape[1]))
                    for lam in dial_lambdas:
                        fdl[lam]["align1"].append(rd[lam]["align1"])
                        fdl[lam]["pred"].append(rd[lam]["pred"])
                except Exception:
                    pass
        got = False
        for a, hs in gk_arms.items():
            for h in hs:
                v = [x for x in fgk[a][h] if x == x]
                if v:
                    acc_gk[a][h].append(float(np.mean(v))); got = True
        for lam in dial_lambdas:
            for key in ("align1", "pred"):
                v = [x for x in fdl[lam][key] if x == x]
                if v:
                    acc_dl[lam][key].append(float(np.mean(v))); got = True
        used += got
    if not used:
        return {"skipped": True}

    out = {"reducer": reducer, "norm": norm, "dims": dims, "n_draws_used": used}
    if do_gkpls:
        for a, hs in gk_arms.items():
            per = {str(h): dict(zip(("spearman_mean", "spearman_std"),
                                    _agg(acc_gk[a][h]))) for h in hs}
            bh = max(per, key=lambda hh: per[hh]["spearman_mean"]
                     if per[hh]["spearman_mean"] == per[hh]["spearman_mean"] else -9)
            out[a] = {"per_hyper": per, "best_hyper": bh, **per[bh]}
    if do_dial:
        dial = {}
        for lam in dial_lambdas:
            a1m, a1s = _agg(acc_dl[lam]["align1"])
            prm, prs = _agg(acc_dl[lam]["pred"])
            dial[str(lam)] = {"align1_mean": a1m, "align1_std": a1s,
                              "pred_mean": prm, "pred_std": prs}
        out["dial"] = dial
    return out
