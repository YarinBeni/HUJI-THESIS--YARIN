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
        self.n_effective: int = 0
        self._iso: IsotonicRegression | None = None
        self._resid: np.ndarray | None = None

    def fit(self, s: np.ndarray, t: np.ndarray,
            groups=None) -> "MonotoneCalibrator":
        """Fit isotonic s->t and bank conformal residuals.

        REVIEW FIX (wave B1): split conformal is only valid when the
        calibration units are exchangeable with the test units. Splitting
        at DOCUMENT level looks valid (measured coverage .800 at nominal
        .80) and is not: t is block-constant within ruler (39 of our 40
        rulers carry ONE year), so documents of the same ruler are near
        copies. Measured leave-one-RULER-out coverage of the doc-split
        calibrator: .665, with 6 of 40 rulers covered 0% of the time —
        the error is all-or-nothing per ruler, not 1-in-5 per document.
        Passing `groups` (the ruler per row) splits BY GROUP, so the
        residual bank comes from rulers the isotonic fit never saw. The
        effective n is then the number of calibration RULERS (~8-12),
        not the number of documents — quote that, not n_docs.
        """
        s = np.asarray(s, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if s.shape != t.shape or len(s) < 4:
            raise ValueError("need matching 1-d s, t with n >= 4")
        rng = np.random.default_rng(self.seed)
        if groups is None:
            order = rng.permutation(len(s))
            n_cal = max(1, int(round(self.calib_frac * len(s))))
            cal, fit_ = order[:n_cal], order[n_cal:]
            self.n_effective = int(n_cal)
        else:
            g = np.asarray(groups).ravel()
            if g.shape != s.shape:
                raise ValueError("groups must align with s")
            uniq = np.array(sorted(set(g.tolist())))
            if len(uniq) < 4:
                raise ValueError("need >= 4 groups for a group split")
            k = max(2, int(round(self.calib_frac * len(uniq))))
            cal_g = set(rng.permutation(uniq)[:k].tolist())
            mask = np.array([x in cal_g for x in g])
            cal, fit_ = np.flatnonzero(mask), np.flatnonzero(~mask)
            self.n_effective = int(k)          # blocks, not documents
        self._iso = IsotonicRegression(increasing=True,
                                       out_of_bounds="clip")
        self._iso.fit(s[fit_], t[fit_])
        err = np.abs(t[cal] - self._iso.predict(s[cal]))
        if groups is None:
            self._resid = np.sort(err)
        else:
            # BLOCK conformal: one residual per calibration RULER (its
            # median document error). Banking per-document residuals
            # would treat ~30 near-copies of one ruler as 30 independent
            # draws and produce intervals that are too narrow for an
            # unseen ruler — measured ruler-coverage .57 at nominal .80.
            gc = np.asarray(groups).ravel()[cal]
            self._resid = np.sort(np.array(
                [np.median(err[gc == u]) for u in sorted(set(gc.tolist()))]))
        return self

    def coverage_by_group(self, s, t, groups, coverage: float = 0.8):
        """Per-group coverage — the number to report for this data.

        Returns (mean over groups, dict group -> coverage). A ruler whose
        every fragment falls outside its interval shows up as 0.0 here
        while the document-level average hides it."""
        lo, hi = self.predict_interval(s, coverage)
        t = np.asarray(t, dtype=float).ravel()
        g = np.asarray(groups).ravel()
        per = {}
        for u in sorted(set(g.tolist())):
            m = g == u
            per[u] = float(((t[m] >= lo[m]) & (t[m] <= hi[m])).mean())
        return float(np.mean(list(per.values()))), per

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
