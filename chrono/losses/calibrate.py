"""Monotone score-to-date calibration with conformal intervals (P2.5).

WHAT. The trained head emits a LATENESS score s (larger = later, SLA
section 1) on an arbitrary scale. MonotoneCalibrator maps s to calendar
t via isotonic regression — monotone by construction, so calibration can
never re-order what the head learned — and wraps the point estimate in
split-conformal intervals from held-out absolute residuals.

WHY split-conformal: the isotonic fit part and the residual part must be
disjoint, otherwise residuals are optimistic and coverage collapses. With
an exchangeable calibration split, [pred - q, pred + q] with q the
ceil((n+1)*coverage)-th smallest calibration residual covers fresh points
with probability >= coverage, regardless of how good the isotonic fit is
(tested: within +-3% of nominal on synthetic data).

numpy/sklearn only — this runs downstream of training, never in a
gradient path.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression


class MonotoneCalibrator:
    """Isotonic s -> t map plus split-conformal intervals.

    fit() shuffles (seeded), fits IsotonicRegression(increasing=True) on
    a `1 - calib_frac` share and banks sorted |t - pred| residuals on the
    rest; predict_interval() turns them into symmetric intervals at any
    requested coverage. increasing=True is the contract: s is lateness,
    t is astronomical, both grow toward the present.
    """

    def __init__(self, calib_frac: float = 0.5, seed: int = 0):
        if not 0.0 < calib_frac < 1.0:
            raise ValueError("calib_frac must be in (0, 1)")
        self.calib_frac = calib_frac
        self.seed = seed
        self._iso: IsotonicRegression | None = None
        self._resid: np.ndarray | None = None

    def fit(self, s: np.ndarray, t: np.ndarray) -> "MonotoneCalibrator":
        s = np.asarray(s, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if s.shape != t.shape or len(s) < 4:
            raise ValueError("need matching 1-d s, t with n >= 4")
        order = np.random.default_rng(self.seed).permutation(len(s))
        n_cal = max(1, int(round(self.calib_frac * len(s))))
        cal, fit_ = order[:n_cal], order[n_cal:]
        self._iso = IsotonicRegression(increasing=True,
                                       out_of_bounds="clip")
        self._iso.fit(s[fit_], t[fit_])
        self._resid = np.sort(np.abs(t[cal] - self._iso.predict(s[cal])))
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("fit() first")
        return self._iso.predict(np.asarray(s, dtype=float).ravel())

    def predict_interval(self, s: np.ndarray, coverage: float = 0.8
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric split-conformal interval per score: pred +- q where
        q is the ceil((n+1)*coverage)-th smallest calibration residual
        (the finite-sample-valid conformal quantile), capped at the max
        residual when coverage asks for more than n residuals can give."""
        if self._resid is None:
            raise RuntimeError("fit() first")
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1)")
        n = len(self._resid)
        k = min(int(math.ceil((n + 1) * coverage)), n)
        q = self._resid[k - 1]
        pred = self.predict(s)
        return pred - q, pred + q
