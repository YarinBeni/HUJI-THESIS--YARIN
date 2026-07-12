"""Monte-Carlo balanced year-probe — the protocol behind the thesis's
maximal-balanced PLS Spearman headline.

For each of the 200 balanced draws (draws_matrix.npy: equal fragments per ruler),
fit GroupKFold-by-ruler within the draw, then average over draws (mean ± std).
Reports BOTH PLS (swept over k, best-k surfaced + full per-k) and Ridge, plus the
shuffled-year null. Same machinery as round2_phase0/run_mc_probes.py; handles
partial-coverage king activations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_probing"))
from pls_utils import fit_pls_groupkfold, fit_ridge_year_groupkfold, l2_normalize  # noqa: E402

PLS_KS = [1, 2, 3, 5]


def _agg(vals):
    v = [x for x in vals if x == x]  # drop NaN
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def mc_year_probe(X, y, g, draw_rows_list, ks=PLS_KS, n_splits=5):
    """X (N,D), y (N,) year, g (N,) ruler. draw_rows_list: per-draw row indices into X.
    Returns dict with best-k PLS (flat keys, back-compat), full per_k PLS, and ridge —
    all averaged over draws (mean/std)."""
    perk = {k: {"sp": [], "r2": [], "mae": [], "shuf": []} for k in ks}
    ridge = {"sp": [], "mae": [], "r2": []}
    used = 0
    for rows in draw_rows_list:
        Xs, ys, gs = X[rows], y[rows], g[rows]
        m = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
        Xs, ys, gs = Xs[m], ys[m], gs[m]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        Xn = l2_normalize(Xs)
        ns = min(n_splits, nr)
        for k in ks:
            if k >= len(Xs):
                continue
            try:
                r = fit_pls_groupkfold(Xn, ys, gs, n_components=k, n_splits=ns)
            except Exception:
                continue
            perk[k]["sp"].append(r["spearman_mean"]); perk[k]["r2"].append(r["r2_mean"])
            perk[k]["mae"].append(r["mae_mean"]); perk[k]["shuf"].append(r.get("shuffled_spearman_mean", np.nan))
        try:
            rr = fit_ridge_year_groupkfold(Xn, ys, np.log(ys), gs, n_splits=ns)["raw"]
            ridge["sp"].append(rr["spearman_mean"]); ridge["mae"].append(rr["mae_mean"])
            ridge["r2"].append(rr["r2_mean"])
        except Exception:
            pass
        used += 1
    if used == 0:
        return {"n_draws_used": 0, "skipped": True}

    per_k = {}
    for k in ks:
        sp_m, sp_s = _agg(perk[k]["sp"])
        per_k[str(k)] = {"spearman_mean": sp_m, "spearman_std": sp_s,
                         "r2_mean": _agg(perk[k]["r2"])[0], "mae_mean": _agg(perk[k]["mae"])[0],
                         "shuffled_spearman_mean": _agg(perk[k]["shuf"])[0]}
    best_k = max(per_k, key=lambda kk: per_k[kk]["spearman_mean"]
                 if per_k[kk]["spearman_mean"] == per_k[kk]["spearman_mean"] else -9)
    rsp_m, rsp_s = _agg(ridge["sp"])
    out = {"n_draws_used": used, "best_k": int(best_k), "per_k": per_k,
           "ridge": {"spearman_mean": rsp_m, "spearman_std": rsp_s,
                     "mae_mean": _agg(ridge["mae"])[0], "r2_mean": _agg(ridge["r2"])[0],
                     "per_draw_spearman": [round(float(x), 4) for x in ridge["sp"]]},
           # per-draw series at best k: draws are SHARED across models, so two
           # runs' series are PAIRED — test differences with
           # eda/significance_mc.py, not by eyeballing mean +- std overlap.
           "per_draw_spearman": [round(float(x), 4) for x in perk[int(best_k)]["sp"]]}
    out.update(per_k[best_k])   # flat best-k PLS keys (back-compat with printers)
    return out


def draws_to_rows(draws_matrix, valid_mask=None):
    """(N_draws, N_corpus) boolean -> list of row-index arrays (positions into the
    full corpus / activation order). If valid_mask (N_corpus, bool) given (king
    found), intersect each draw with it."""
    out = []
    for d in range(draws_matrix.shape[0]):
        rows = np.where(draws_matrix[d])[0]
        if valid_mask is not None:
            rows = rows[valid_mask[rows]]
        out.append(rows)
    return out
