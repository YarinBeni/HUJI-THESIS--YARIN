"""Monte-Carlo probe engine for the "maximal-with-kings" config.

Because the ORCC `year` label is a single constant per king (see eda/), decoding
year is essentially the same task as identifying the ruler. So for each balanced
draw we run THREE analyses on the pooled activations and average over draws:

  1. year_group  — PLS (k-swept) + Ridge under GroupKFold-BY-RULER. This is the
                   legacy balanced-MC headline protocol (extrapolate to a held-out
                   king). With a per-king-constant label it is largely degenerate
                   (Spearman undefined on single-year folds); kept for continuity.
                   n_splits capped at 2 so test folds contain >= 2 rulers.
  2. year_strat  — PLS (k-swept) + Ridge under StratifiedKFold-BY-RULER (rulers
                   mixed across folds). In-distribution "can you read the date";
                   reports Spearman, MAE and within-+/-10yr accuracy.
  3. ruler_clf   — PLS-DA (k-swept) under StratifiedKFold: the CONTROL. macro-F1 /
                   accuracy of predicting the ruler label, vs chance and a shuffle
                   baseline. If a RANDOM-weights model matches a trained one here,
                   the site is reading name-token identity, not learned structure.

Each task sweeps k in {1,2,3,5}; the best-k (by the task's headline metric,
averaged over draws) is surfaced with the full per-k table retained.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_probing"))
from pls_utils import (fit_pls_groupkfold, fit_ridge_year_groupkfold,  # noqa: E402
                       fit_plsda_stratified_kfold, l2_normalize)

PLS_KS = [1, 2, 3, 5]
ACC_TOL = 10  # years


def _mean(vals):
    v = [x for x in vals if x == x]
    return float(np.mean(v)) if v else float("nan")


def _std(vals):
    v = [x for x in vals if x == x]
    return float(np.std(v)) if v else float("nan")


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(np.asarray(a).tolist())) < 2:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def _year_strat(Xn, y, strat, k, n_splits):
    """PLS year regression under StratifiedKFold(by ruler). Returns sp, mae, acc10."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.cross_decomposition import PLSRegression
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sp, mae, acc = [], [], []
    for tr, te in skf.split(Xn, strat):
        if k >= len(tr):
            continue
        m = PLSRegression(n_components=k).fit(Xn[tr], y[tr])
        yp = m.predict(Xn[te]).ravel()
        sp.append(_spearman(y[te], yp))
        mae.append(float(np.mean(np.abs(yp - y[te]))))
        acc.append(float(np.mean(np.abs(yp - y[te]) <= ACC_TOL)))
    return _mean(sp), _mean(mae), _mean(acc)


def _year_strat_ridge(Xn, y, strat, n_splits):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import Ridge
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sp, mae, acc = [], [], []
    for tr, te in skf.split(Xn, strat):
        m = Ridge(alpha=10.0).fit(Xn[tr], y[tr])
        yp = m.predict(Xn[te])
        sp.append(_spearman(y[te], yp))
        mae.append(float(np.mean(np.abs(yp - y[te]))))
        acc.append(float(np.mean(np.abs(yp - y[te]) <= ACC_TOL)))
    return _mean(sp), _mean(mae), _mean(acc)


def mc_maxking_probe(X, year, ruler, draw_rows_list, ks=PLS_KS, n_splits=5):
    """X (N,D) activations; year (N,), ruler (N,) aligned to corpus order.
    draw_rows_list: per-draw row indices (already king-found for king sites)."""
    yg = {k: {"sp": [], "mae": []} for k in ks}            # year_group PLS per-k
    yg_ridge = {"sp": [], "mae": []}
    ys = {k: {"sp": [], "mae": [], "acc": []} for k in ks}  # year_strat PLS per-k
    ys_ridge = {"sp": [], "mae": [], "acc": []}
    rc = {k: {"f1": [], "acc": []} for k in ks}             # ruler_clf per-k
    rc_meta = {"chance_macro_f1": [], "chance_accuracy": [], "shuffled_macro_f1": []}
    used = 0

    for rows in draw_rows_list:
        Xs, ysr, gs = X[rows], year[rows], ruler[rows]
        m = np.isfinite(Xs).all(axis=1) & np.isfinite(ysr)
        Xs, ysr, gs = Xs[m], ysr[m], gs[m]
        nr = len(set(gs.tolist()))
        if len(Xs) < 10 or nr < 2:
            continue
        Xn = l2_normalize(Xs)
        # class counts limit StratifiedKFold splits
        _, cnts = np.unique(gs, return_counts=True)
        ns_strat = int(max(2, min(n_splits, cnts.min())))
        ns_group = int(min(2, nr))  # >=2 rulers per test fold for a defined Spearman

        for k in ks:
            if k >= len(Xs):
                continue
            # 1. year_group PLS (GroupKFold by ruler)
            try:
                r = fit_pls_groupkfold(Xn, ysr, gs, n_components=k, n_splits=ns_group)
                yg[k]["sp"].append(r["spearman_mean"]); yg[k]["mae"].append(r["mae_mean"])
            except Exception:
                pass
            # 2. year_strat PLS
            sp, mae, acc = _year_strat(Xn, ysr, gs, k, ns_strat)
            ys[k]["sp"].append(sp); ys[k]["mae"].append(mae); ys[k]["acc"].append(acc)
            # 3. ruler_clf PLS-DA
            try:
                c = fit_plsda_stratified_kfold(Xn, gs, n_components=k, n_splits=ns_strat)
                rc[k]["f1"].append(c["macro_f1_mean"]); rc[k]["acc"].append(c["accuracy_mean"])
                if k == ks[0]:
                    rc_meta["chance_macro_f1"].append(c["chance_macro_f1"])
                    rc_meta["chance_accuracy"].append(c["chance_accuracy"])
                    rc_meta["shuffled_macro_f1"].append(c["shuffled_macro_f1_mean"])
            except Exception:
                pass
        # ridge arms (single, no k)
        try:
            rr = fit_ridge_year_groupkfold(Xn, ysr, np.log(ysr), gs, n_splits=ns_group)["raw"]
            yg_ridge["sp"].append(rr["spearman_mean"]); yg_ridge["mae"].append(rr["mae_mean"])
        except Exception:
            pass
        sp, mae, acc = _year_strat_ridge(Xn, ysr, gs, ns_strat)
        ys_ridge["sp"].append(sp); ys_ridge["mae"].append(mae); ys_ridge["acc"].append(acc)
        used += 1

    if used == 0:
        return {"n_draws_used": 0, "skipped": True}

    def per_k(store, metric):
        return {str(k): {"mean": _mean(store[k][metric]), "std": _std(store[k][metric])} for k in ks}

    yg_perk = {str(k): {"spearman_mean": _mean(yg[k]["sp"]), "spearman_std": _std(yg[k]["sp"]),
                        "mae_mean": _mean(yg[k]["mae"])} for k in ks}
    ys_perk = {str(k): {"spearman_mean": _mean(ys[k]["sp"]), "spearman_std": _std(ys[k]["sp"]),
                        "mae_mean": _mean(ys[k]["mae"]), "acc10_mean": _mean(ys[k]["acc"])} for k in ks}
    rc_perk = {str(k): {"macro_f1_mean": _mean(rc[k]["f1"]), "macro_f1_std": _std(rc[k]["f1"]),
                        "accuracy_mean": _mean(rc[k]["acc"])} for k in ks}

    def best_k(perk, key):
        return max(perk, key=lambda kk: perk[kk][key] if perk[kk][key] == perk[kk][key] else -9)

    bg, bs, br = best_k(yg_perk, "spearman_mean"), best_k(ys_perk, "spearman_mean"), best_k(rc_perk, "macro_f1_mean")
    return {
        "n_draws_used": used,
        "year_group": {"best_k": int(bg), "per_k": yg_perk, **yg_perk[bg],
                       "ridge": {"spearman_mean": _mean(yg_ridge["sp"]), "mae_mean": _mean(yg_ridge["mae"])}},
        "year_strat": {"best_k": int(bs), "per_k": ys_perk, **ys_perk[bs],
                       "ridge": {"spearman_mean": _mean(ys_ridge["sp"]), "mae_mean": _mean(ys_ridge["mae"]),
                                 "acc10_mean": _mean(ys_ridge["acc"])}},
        "ruler_clf": {"best_k": int(br), "per_k": rc_perk, **rc_perk[br],
                      "chance_macro_f1": _mean(rc_meta["chance_macro_f1"]),
                      "chance_accuracy": _mean(rc_meta["chance_accuracy"]),
                      "shuffled_macro_f1": _mean(rc_meta["shuffled_macro_f1"])},
    }
