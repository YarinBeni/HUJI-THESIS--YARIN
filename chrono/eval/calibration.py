"""Interval calibration metrics — does a dated interval actually cover?

WHAT. Given per-doc predictive intervals [lo, hi] for composition year
t (astronomical, larger = later, SLA §1), the numbers a calibrated
dater must report: empirical coverage (share of true t inside its
interval, bounds inclusive — the split-conformal convention), mean
interval width (what that coverage costs), and the Winkler interval
score, which trades the two off at a stated nominal coverage (width
plus 2/alpha times any overshoot; lower is better, and an interval can
no longer win by being vacuously wide or dishonestly narrow).

WHY here and not in the calibrator. A3's MonotoneCalibrator PRODUCES
intervals; whether they cover is an evaluation claim, and per SLA §7
every evaluation number comes from chrono/eval so the module under
evaluation can never grade itself.

Pure numpy; no torch.
"""
from __future__ import annotations

import numpy as np

__all__ = ["coverage", "mean_width", "winkler_score"]


def _intervals(lo, hi) -> tuple:
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if lo.shape != hi.shape:
        raise ValueError(f"lo/hi shape mismatch: {lo.shape} vs {hi.shape}")
    if lo.size == 0:
        raise ValueError("empty intervals")
    if np.any(hi < lo):
        k = int(np.argmax(hi < lo))
        raise ValueError(f"hi < lo at position {k}: "
                         f"[{lo[k]:g}, {hi[k]:g}]")
    return lo, hi


def _with_t(lo, hi, t) -> tuple:
    lo, hi = _intervals(lo, hi)
    t = np.asarray(t, dtype=float).ravel()
    if t.shape != lo.shape:
        raise ValueError(f"t shape {t.shape} != intervals {lo.shape}")
    return lo, hi, t


def coverage(lo, hi, t) -> float:
    """Fraction of true t with lo <= t <= hi (bounds inclusive)."""
    lo, hi, t = _with_t(lo, hi, t)
    return float(np.mean((t >= lo) & (t <= hi)))


def mean_width(lo, hi) -> float:
    """Mean interval width hi - lo (same units as t: years)."""
    lo, hi = _intervals(lo, hi)
    return float(np.mean(hi - lo))


def winkler_score(lo, hi, t, *, nominal: float = 0.8) -> float:
    """Mean Winkler score of central (nominal)-coverage intervals.

    Per doc: (hi - lo) + (2/alpha) * max(lo - t, 0)
                       + (2/alpha) * max(t - hi, 0),  alpha = 1 - nominal.
    Lower is better; proper for central prediction intervals.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal must be in (0, 1), got {nominal}")
    lo, hi, t = _with_t(lo, hi, t)
    alpha = 1.0 - nominal
    miss = np.maximum(lo - t, 0.0) + np.maximum(t - hi, 0.0)
    return float(np.mean((hi - lo) + (2.0 / alpha) * miss))
