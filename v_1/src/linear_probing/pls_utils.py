"""
Shared PLS utilities for the linear probing pipeline.

Public API
----------
l2_normalize(X)                                              -> np.ndarray
compute_metrics(y_true, y_pred, y_train_for_mase)            -> dict (r2,spearman,mae,mase,mdape)
fit_pls_groupkfold(X, y, groups, n_components, n_splits=5,
                   random_state=42)                          -> dict (mean/std/folds + shuffled baseline)
fit_pls_full(X, y, n_components=5)                           -> PLSRegression
project(model, X)                                            -> (N, n_components) np.ndarray
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr as _spearmanr


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row: xi / max(||xi||_2, eps)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def _spearman(a, b) -> float:
    res = _spearmanr(a, b)
    return float(res.statistic if hasattr(res, 'statistic') else res[0])


def compute_metrics(y_true, y_pred, y_train_for_mase) -> dict:
    """
    Compute regression metrics for one prediction set.

    mase  = mae / mean(|y_train - mean(y_train)|)
    mdape = median(|pred - true| / max(|true|, 1)) * 100

    Returns dict with keys: r2, spearman, mae, mase, mdape.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_tr   = np.asarray(y_train_for_mase, dtype=float)

    mae   = float(np.mean(np.abs(y_true - y_pred)))
    naive = float(np.mean(np.abs(y_tr - np.mean(y_tr))))
    mase  = mae / max(naive, 1e-10)
    mdape = float(
        np.median(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1.0)) * 100
    )
    r2 = float(r2_score(y_true, y_pred))
    sp = _spearman(y_true, y_pred)

    return {'r2': r2, 'spearman': sp, 'mae': mae, 'mase': mase, 'mdape': mdape}


# ---------------------------------------------------------------------------
# Shuffled baseline helper
# ---------------------------------------------------------------------------

def _shuffle_within_groups(y: np.ndarray, groups: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """Permute y values independently within each unique group."""
    y_s = y.copy()
    for g in np.unique(groups):
        mask = groups == g
        y_s[mask] = rng.permutation(y_s[mask])
    return y_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pls_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups,
    n_components: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    GroupKFold CV with PLSRegression(n_components=n_components).

    Returns a dict matching the results JSON schema for one k value:
      {r2_mean, r2_std, r2_folds, spearman_mean, ..., shuffled_r2_mean, shuffled_spearman_mean}
    """
    groups = np.asarray(groups)
    y      = np.asarray(y, dtype=float)
    gkf    = GroupKFold(n_splits=n_splits)
    keys   = ['r2', 'spearman', 'mae', 'mase', 'mdape']
    folds  = {k: [] for k in keys}

    splits = list(gkf.split(X, y, groups))

    for tr_idx, val_idx in splits:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr_idx], y[tr_idx])
        m = compute_metrics(y[val_idx], pls.predict(X[val_idx]).ravel(), y[tr_idx])
        for k in keys:
            folds[k].append(m[k])

    # Shuffled-y baseline: permute y within ruler groups, same CV splits
    rng = np.random.default_rng(random_state)
    shuf_r2, shuf_sp = [], []
    for tr_idx, val_idx in splits:
        y_s = _shuffle_within_groups(y, groups, rng)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr_idx], y_s[tr_idx])
        y_pred_s = pls.predict(X[val_idx]).ravel()
        shuf_r2.append(float(r2_score(y_s[val_idx], y_pred_s)))
        shuf_sp.append(_spearman(y_s[val_idx], y_pred_s))

    # A fold is degenerate when y_test is constant → Spearman = NaN.
    # Use Spearman NaN status to identify degenerate folds; apply same mask to
    # all metrics so means are over the same set of folds.
    sp_vals = folds['spearman']
    valid_mask = [i for i, v in enumerate(sp_vals) if not np.isnan(v)]
    n_valid = len(valid_mask)
    n_total = len(sp_vals)

    result = {'n_valid_folds': n_valid, 'n_total_folds': n_total}
    for k in keys:
        vals = folds[k]
        valid_vals = [vals[i] for i in valid_mask]
        result[f'{k}_mean']  = float(np.nanmean(valid_vals)) if valid_vals else float('nan')
        result[f'{k}_std']   = float(np.nanstd(valid_vals)) if valid_vals else float('nan')
        result[f'{k}_folds'] = [float(v) for v in vals]

    valid_shuf_r2 = [shuf_r2[i] for i in valid_mask]
    valid_shuf_sp = [shuf_sp[i] for i in valid_mask]
    result['shuffled_r2_mean']       = float(np.nanmean(valid_shuf_r2)) if valid_shuf_r2 else float('nan')
    result['shuffled_spearman_mean'] = float(np.nanmean(valid_shuf_sp)) if valid_shuf_sp else float('nan')
    return result


def fit_pls_full(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 5,
) -> PLSRegression:
    """Fit PLSRegression on the full labeled set (no holdout)."""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, np.asarray(y, dtype=float))
    return pls


def project(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Project X onto PLS latent space. Returns (N, n_components) array."""
    return model.transform(X)
