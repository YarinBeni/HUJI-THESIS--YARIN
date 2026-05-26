"""
Shared PLS utilities for the linear probing pipeline.

Public API
----------
l2_normalize(X)                                              -> np.ndarray
compute_metrics(y_true, y_pred, y_train_for_mase)            -> dict (r2,spearman,mae,mase,mdape)
fit_pls_groupkfold(X, y, groups, n_components, n_splits=5,
                   random_state=42)                          -> dict (mean/std/folds + shuffled baseline)
fit_pls_full(X, y, n_components=5)                           -> PLSRegression
fit_plsda_stratified_kfold(X, y, n_components, n_splits=5,
                            random_state=42)                 -> dict (accuracy/f1 + baselines)
fit_plsda_full(X, y, n_components=5)                         -> PLSRegression (fitted on one-hot y)
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

def _global_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Globally permute all y labels (true null distribution for regression)."""
    return rng.permutation(y)


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

    # Shuffled-y baseline: globally permute y (true null), same CV splits
    rng = np.random.default_rng(random_state)
    shuf_r2, shuf_sp = [], []
    for tr_idx, val_idx in splits:
        y_s = _global_shuffle(y, rng)
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
    result['shuffled_r2_folds']       = [float(v) for v in shuf_r2]
    result['shuffled_spearman_folds'] = [float(v) for v in shuf_sp]
    return result


def fit_plsda_stratified_kfold(
    X: np.ndarray,
    y,
    n_components: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    StratifiedKFold PLS-DA for categorical targets (e.g. ruler, 38 classes).

    Encodes y as a one-hot matrix (N x n_classes), fits PLSRegression,
    predicts by argmax. Returns accuracy/macro_f1/weighted_f1 mean+std+folds
    and chance baselines plus global-shuffle baseline.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    from collections import Counter

    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    n_classes = len(le.classes_)

    Y_oh = np.zeros((len(y_enc), n_classes), dtype=float)
    Y_oh[np.arange(len(y_enc)), y_enc] = 1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y_enc))

    accs, mac_f1s, wt_f1s = [], [], []
    for tr_idx, val_idx in splits:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr_idx], Y_oh[tr_idx])
        pred_oh = pls.predict(X[val_idx])
        yp = np.argmax(pred_oh, axis=1)
        accs.append(float(accuracy_score(y_enc[val_idx], yp)))
        mac_f1s.append(float(f1_score(y_enc[val_idx], yp, average='macro', zero_division=0)))
        wt_f1s.append(float(f1_score(y_enc[val_idx], yp, average='weighted', zero_division=0)))

    rng = np.random.default_rng(random_state)
    shuf_accs, shuf_mac_f1s = [], []
    for tr_idx, val_idx in splits:
        y_s = rng.permutation(y_enc)
        Y_s_oh = np.zeros_like(Y_oh)
        Y_s_oh[np.arange(len(y_s)), y_s] = 1.0
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr_idx], Y_s_oh[tr_idx])
        pred_oh = pls.predict(X[val_idx])
        yp = np.argmax(pred_oh, axis=1)
        shuf_accs.append(float(accuracy_score(y_s[val_idx], yp)))
        shuf_mac_f1s.append(float(f1_score(y_s[val_idx], yp, average='macro', zero_division=0)))

    counts = Counter(y_enc.tolist())
    majority_frac = max(counts.values()) / len(y_enc)

    return {
        'n_classes':               n_classes,
        'n_splits':                n_splits,
        'chance_accuracy':         float(majority_frac),
        'chance_macro_f1':         float(1.0 / n_classes),
        'accuracy_mean':           float(np.mean(accs)),
        'accuracy_std':            float(np.std(accs)),
        'accuracy_folds':          accs,
        'macro_f1_mean':           float(np.mean(mac_f1s)),
        'macro_f1_std':            float(np.std(mac_f1s)),
        'macro_f1_folds':          mac_f1s,
        'weighted_f1_mean':        float(np.mean(wt_f1s)),
        'weighted_f1_std':         float(np.std(wt_f1s)),
        'weighted_f1_folds':       wt_f1s,
        'shuffled_accuracy_mean':  float(np.mean(shuf_accs)),
        'shuffled_macro_f1_mean':  float(np.mean(shuf_mac_f1s)),
    }


def fit_plsda_full(X: np.ndarray, y, n_components: int = 5) -> PLSRegression:
    """Fit PLS-DA on full labeled set. y = categorical labels (strings or ints)."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    n_classes = len(le.classes_)
    Y_oh = np.zeros((len(y_enc), n_classes), dtype=float)
    Y_oh[np.arange(len(y_enc)), y_enc] = 1.0
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y_oh)
    return pls


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


def fit_ridge_year_groupkfold(
    X: np.ndarray,
    y_raw: np.ndarray,
    y_log: np.ndarray,
    groups,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> dict:
    """Ridge regression for year (cls_numeric probe).

    GroupKFold CV with ruler as group — same split strategy as PLS so results
    are directly comparable. Evaluates in raw-year space for both transforms
    (log-predicted values are back-transformed via exp before scoring).

    Returns a dict with keys 'raw' and 'log', each containing:
      spearman_mean/std, mae_mean/std, r2_mean/std, mase_mean/std,
      mdape_mean/std, shuffled_spearman_mean, shuffled_r2_mean
    """
    from sklearn.linear_model import Ridge

    gkf = GroupKFold(n_splits=n_splits)
    groups = np.asarray(groups)

    results: dict = {}
    for yt, y in [("raw", np.asarray(y_raw, dtype=float)),
                  ("log", np.asarray(y_log, dtype=float))]:
        spearmans, maes, r2s, mases, mdapes = [], [], [], [], []
        shuf_sp, shuf_r2 = [], []
        # Shuffled-y baseline: globally permute y once per fit (true null),
        # mirroring fit_pls_groupkfold. Seeded for determinism.
        rng = np.random.default_rng(42)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            model = Ridge(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            if yt == "log":
                pred_y     = np.exp(np.clip(pred, -30, 30))
                true_y     = np.exp(y[test_idx])
                train_eval = np.exp(y[train_idx])
            else:
                pred_y     = pred
                true_y     = y[test_idx]
                train_eval = y[train_idx]
            spearmans.append(_spearman(true_y, pred_y))
            mae_fold = float(np.mean(np.abs(true_y - pred_y)))
            maes.append(mae_fold)
            ss_res = float(np.sum((true_y - pred_y) ** 2))
            ss_tot = float(np.sum((true_y - np.mean(true_y)) ** 2))
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            # MASE: MAE relative to the naive in-sample mean predictor.
            denom = max(float(np.mean(np.abs(train_eval - np.mean(train_eval)))), 1e-10)
            mases.append(mae_fold / denom)
            # MdAPE: median absolute percentage error (%).
            mdapes.append(float(np.median(
                np.abs(pred_y - true_y) / np.maximum(np.abs(true_y), 1.0))) * 100.0)
            # Shuffled baseline: permute y globally, refit, back-transform if log.
            y_s = rng.permutation(y)
            sm = Ridge(alpha=alpha)
            sm.fit(X[train_idx], y_s[train_idx])
            spred = sm.predict(X[test_idx])
            if yt == "log":
                spred_y = np.exp(np.clip(spred, -30, 30))
                strue_y = np.exp(y_s[test_idx])
            else:
                spred_y = spred
                strue_y = y_s[test_idx]
            shuf_sp.append(_spearman(strue_y, spred_y))
            s_ss_res = float(np.sum((strue_y - spred_y) ** 2))
            s_ss_tot = float(np.sum((strue_y - np.mean(strue_y)) ** 2))
            shuf_r2.append(1.0 - s_ss_res / s_ss_tot if s_ss_tot > 0 else 0.0)
        # Mirror fit_pls_groupkfold: exclude folds where y_test is constant
        # (one unique year in fold → Spearman = NaN). Use same mask for all metrics.
        valid = [i for i, v in enumerate(spearmans) if not np.isnan(v)]
        def _vmean(lst): return float(np.nanmean([lst[i] for i in valid])) if valid else float("nan")
        def _vstd(lst):  return float(np.nanstd( [lst[i] for i in valid])) if valid else float("nan")
        results[yt] = {
            "spearman_mean":          _vmean(spearmans),
            "spearman_std":           _vstd(spearmans),
            "mae_mean":               _vmean(maes),
            "mae_std":                _vstd(maes),
            "r2_mean":                _vmean(r2s),
            "r2_std":                 _vstd(r2s),
            "mase_mean":              _vmean(mases),
            "mase_std":               _vstd(mases),
            "mdape_mean":             _vmean(mdapes),
            "mdape_std":              _vstd(mdapes),
            "shuffled_spearman_mean": _vmean(shuf_sp),
            "shuffled_r2_mean":       _vmean(shuf_r2),
            "n_valid_folds":          len(valid),
            "n_total_folds":          len(spearmans),
        }
    return results
