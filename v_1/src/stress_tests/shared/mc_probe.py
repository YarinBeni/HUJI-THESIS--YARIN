"""Monte-Carlo balanced year-probe — the protocol behind the thesis's
maximal-balanced PLS Spearman headline.

For each of the 200 balanced draws (draws_matrix.npy: equal fragments per ruler),
fit GroupKFold-by-ruler PLS within the draw, take the best-k Spearman, then
average over draws (mean ± std). Same machinery as
round2_phase0/run_mc_probes.py, exposed as a helper so the stress-test probes
(P1 king sites, T10) can use the identical protocol on partial-coverage king
activations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_probing"))
from pls_utils import fit_pls_groupkfold, l2_normalize  # noqa: E402

PLS_KS = [1, 2, 3, 5]


def mc_year_probe(X, y, g, draw_rows_list, ks=PLS_KS, n_splits=5):
    """X (N,D) features, y (N,) year, g (N,) ruler groups. draw_rows_list: list of
    arrays of ROW INDICES into X (one per draw). Returns aggregate dict
    (mean/std over draws of the best-k-per-draw Spearman + r2/mae + shuffled null)."""
    sp, r2, mae, shuf, used = [], [], [], [], 0
    for rows in draw_rows_list:
        Xs, ys, gs = X[rows], y[rows], g[rows]
        m = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        Xs, ys, gs = Xs[m], ys[m], gs[m]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        Xn = l2_normalize(Xs)
        best, best_r = -9.0, None
        for k in ks:
            if k >= len(Xs):
                continue
            try:
                r = fit_pls_groupkfold(Xn, ys, gs, n_components=k, n_splits=min(n_splits, nr))
            except Exception:
                continue
            s = r["spearman_mean"]
            if s == s and s > best:   # not NaN and better
                best, best_r = s, r
        if best_r is not None:
            sp.append(best_r["spearman_mean"]); r2.append(best_r["r2_mean"])
            mae.append(best_r["mae_mean"]); shuf.append(best_r.get("shuffled_spearman_mean", np.nan))
            used += 1
    if used == 0:
        return {"n_draws_used": 0, "skipped": True}
    f = lambda a: float(np.nanmean(a))
    s = lambda a: float(np.nanstd(a))
    return {"n_draws_used": used,
            "spearman_mean": f(sp), "spearman_std": s(sp),
            "r2_mean": f(r2), "mae_mean": f(mae), "shuffled_spearman_mean": f(shuf)}


def draws_to_rows(draws_matrix, valid_mask=None):
    """Convert the (N_draws, N_corpus) boolean matrix into a list of row-index
    arrays (positions into the full corpus / activation order). If valid_mask
    (N_corpus, bool) is given (e.g. king-name found), intersect each draw with it."""
    out = []
    for d in range(draws_matrix.shape[0]):
        rows = np.where(draws_matrix[d])[0]
        if valid_mask is not None:
            rows = rows[valid_mask[rows]]
        out.append(rows)
    return out
