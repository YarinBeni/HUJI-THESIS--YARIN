"""Stress-test metrics. Reuses the project's PLS metrics and adds the two metrics
the mirrored papers emphasize: Gurnee-Tegmark's *proximity error* (a 1-D rank
error for time) and a great-circle distance for the P2 geography mirror.
"""
from __future__ import annotations

import numpy as np

# Reuse the canonical regression metric block (r2, spearman, mae, mase, mdape).
try:  # importable when run from v_1/src/linear_probing on path
    from pls_utils import compute_metrics  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback path resolution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_probing"))
    from pls_utils import compute_metrics  # type: ignore  # noqa: F401


def proximity_error(y_true, y_pred):
    """Gurnee-Tegmark proximity error, 1-D (time) version: for each test point,
    the fraction of OTHER true points that the prediction lands closer to than the
    point's own true value. 0 = perfect localization, ~0.5 = chance. Returns the
    mean over points."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    if n < 2:
        return float("nan")
    out = np.empty(n)
    for i in range(n):
        d_pred = np.abs(yt - yp[i])          # how close pred i is to every true point
        d_self = abs(yt[i] - yp[i])          # how close pred i is to its own true point
        others = np.delete(d_pred < d_self, i)
        out[i] = others.mean()
    return float(out.mean())


def great_circle_km(lat1, lon1, lat2, lon2, radius_km: float = 6371.0):
    """Haversine distance (km). Accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                              np.asarray(lat2, float), np.asarray(lon2, float)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(radius_km) * 2 * np.arcsin(np.sqrt(a))


def geo_metrics(lat_true, lon_true, lat_pred, lon_pred):
    """Mean/median great-circle error (km) + spatial proximity error."""
    err = great_circle_km(lat_true, lon_true, lat_pred, lon_pred)
    err = np.atleast_1d(err)
    lat_true = np.asarray(lat_true, float); lon_true = np.asarray(lon_true, float)
    lat_pred = np.asarray(lat_pred, float); lon_pred = np.asarray(lon_pred, float)
    n = len(lat_true)
    prox = np.full(n, np.nan)
    if n >= 2:
        for i in range(n):
            d_pred = great_circle_km(lat_true, lon_true, lat_pred[i], lon_pred[i])
            d_self = great_circle_km(lat_true[i], lon_true[i], lat_pred[i], lon_pred[i])
            prox[i] = np.delete(d_pred < d_self, i).mean()
    return {"gc_km_mean": float(err.mean()), "gc_km_median": float(np.median(err)),
            "proximity_error": float(np.nanmean(prox))}
