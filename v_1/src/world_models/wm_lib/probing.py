"""Linear probes + scoring, ported from wesg52/world-models
(probe_experiment.py, probes/evaluation.py) with two corrections/notes:

* haversine gets (lat, lon) explicitly (their coords target is (lon, lat); we keep
  that column order for per-axis scores but swap before the haversine call).
* RidgeCV alpha grid np.logspace(-1, 5, 13): superset of their per-model grids
  (10^0.8..10^4.5), needed because our arms span d=768..8192.

The probe contract everywhere: fit on ~is_test rows, score on both splits, targets
z-scored on train stats, predictions un-normalized before scoring.
"""
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Ridge, RidgeCV

ALPHAS = np.logspace(-1, 5, 13)


# ---- distances / scores (verbatim ports unless noted) -----------------------


FP16_MAX = 65504.0


def sanitize(X):
    """Activations are stored as fp16, whose range is +/-65504. Models with very large
    outlier activations (gpt-oss-120B especially) overflow that range and land on disk
    as +/-inf, which sklearn rejects. Clamp non-finite entries back to the representable
    bound and report how much was affected so the caller can drop a badly corrupt layer.

    Returns (X_clean float32, fraction_non_finite)."""
    X = np.asarray(X)
    bad = ~np.isfinite(X)
    frac = float(bad.mean()) if bad.size else 0.0
    if frac:
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0,
                          posinf=FP16_MAX, neginf=-FP16_MAX)
    return X, frac

def haversine_distance(true_latlon, pred_latlon):
    """(n,2) arrays in (lat, lon) order -> km."""
    R = 6371.0
    t = np.radians(np.asarray(true_latlon, dtype=np.float64))
    p = np.radians(np.asarray(pred_latlon, dtype=np.float64))
    dlat = t[:, 0] - p[:, 0]
    dlon = t[:, 1] - p[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(p[:, 0]) * np.cos(t[:, 0]) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def haversine_r2(true_latlon, pred_latlon):
    mean_coords = np.mean(true_latlon, axis=0)
    ss_tot = np.sum(haversine_distance(
        true_latlon, np.repeat(mean_coords[None, :], len(true_latlon), 0)) ** 2)
    ss_res = np.sum(haversine_distance(true_latlon, pred_latlon) ** 2)
    return 1 - ss_res / ss_tot


def score_place(target_lonlat, pred_lonlat):
    """target/pred in (lon, lat) column order, like their coords target."""
    t, p = np.asarray(target_lonlat), np.asarray(pred_lonlat)
    sd = {
        "r2": metrics.r2_score(t, p),
        "lon_r2": metrics.r2_score(t[:, 0], p[:, 0]),
        "lat_r2": metrics.r2_score(t[:, 1], p[:, 1]),
        "mae": metrics.mean_absolute_error(t, p),
        "lon_spearman": stats.spearmanr(t[:, 0], p[:, 0]).correlation,
        "lat_spearman": stats.spearmanr(t[:, 1], p[:, 1]).correlation,
        "lon_pearson": stats.pearsonr(t[:, 0], p[:, 0])[0],
        "lat_pearson": stats.pearsonr(t[:, 1], p[:, 1])[0],
    }
    hav = haversine_distance(t[:, ::-1], p[:, ::-1])   # swap to (lat, lon)
    sd["haversine_mae"] = float(np.mean(hav))
    sd["haversine_rmse"] = float(np.sqrt(np.mean(hav ** 2)))
    sd["haversine_r2"] = haversine_r2(t[:, ::-1], p[:, ::-1])
    return {k: float(v) for k, v in sd.items()}


def score_time(target, pred):
    t, p = np.asarray(target).ravel(), np.asarray(pred).ravel()
    return {
        "r2": float(metrics.r2_score(t, p)),
        "mae": float(metrics.mean_absolute_error(t, p)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(t, p))),
        "pearson": float(stats.pearsonr(t, p)[0]),
        "spearman": float(stats.spearmanr(t, p).correlation),
        "kendall": float(stats.kendalltau(t, p).correlation),
    }


# ---- the probe experiment ---------------------------------------------------

def run_probe(activations, target, is_test, is_place, probe=None):
    """One layer's probe. activations (n,d) float; target (n,) or (n,2);
    is_test bool (n,). Returns (scores dict, probe, projection (n,) or (n,2))."""
    X = np.asarray(activations, dtype=np.float32)
    y = np.asarray(target, dtype=np.float64)
    tr, te = ~is_test, is_test

    mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
    y_norm = (y[tr] - mu) / sd

    if probe is None:
        probe = RidgeCV(alphas=ALPHAS)
    probe.fit(X[tr], y_norm)

    proj = probe.predict(X) * sd + mu
    score_fn = score_place if is_place else score_time
    scores = {
        "train": score_fn(y[tr], proj[tr]),
        "test": score_fn(y[te], proj[te]),
    }
    if hasattr(probe, "alpha_"):
        scores["alpha"] = float(np.atleast_1d(probe.alpha_)[0])
    return scores, probe, proj


def run_pls_probe(activations, target, is_test, is_place, k=5):
    """Thesis-canonical PLS probe (secondary, --probe pls)."""
    from sklearn.cross_decomposition import PLSRegression
    X = np.asarray(activations, dtype=np.float32)
    y = np.asarray(target, dtype=np.float64)
    tr, te = ~is_test, is_test
    mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
    pls = PLSRegression(n_components=min(k, X.shape[1] - 1), scale=False)
    pls.fit(X[tr], (y[tr] - mu) / sd)
    proj = pls.predict(X)
    if proj.ndim == 2 and np.ndim(target) == 1:
        proj = proj.ravel()
    proj = proj * sd + mu
    score_fn = score_place if is_place else score_time
    return {
        "train": score_fn(y[tr], proj[tr]),
        "test": score_fn(y[te], proj[te]),
        "k": int(k),
    }, pls, proj
