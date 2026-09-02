"""LEACE (Belrose et al., NeurIPS 2023) in closed form, numpy only.

r(x) = x - W^+ P_{W Σ_xz} W (x - μ_x),   W = Σ_xx^{-1/2} (whitening)

i.e. project out, in whitened space, the span of the cross-covariance
between X and the one-hot concept Z; this is the least-squares-optimal
linear map after which NO linear probe can read Z (E[Z|linear(X)] = E[Z]),
while moving X as little as possible. Phase 2 used the `concept_erasure`
package; this is the same estimator without the dependency, fitted on the
TRAIN fold and applied to train and test (never fitted on held-out docs).
"""
from __future__ import annotations

import numpy as np

__all__ = ["LeaceEraser", "z_readability"]


class LeaceEraser:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "LeaceEraser":
        X = np.asarray(X, np.float64); Z = np.asarray(Z, np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        self.mu_x = X.mean(0); self.mu_z = Z.mean(0)
        Xc, Zc = X - self.mu_x, Z - self.mu_z
        n = len(X)
        sxx = Xc.T @ Xc / n + self.eps * np.eye(X.shape[1])
        sxz = Xc.T @ Zc / n
        # whitening / colouring through the eigendecomposition of Σ_xx
        w, V = np.linalg.eigh(sxx)
        w = np.clip(w, self.eps, None)
        W = V @ np.diag(w ** -0.5) @ V.T          # Σ^{-1/2}
        Winv = V @ np.diag(w ** 0.5) @ V.T        # Σ^{+1/2}
        A = W @ sxz                                # whitened cross-cov, d x k
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        keep = s > (s.max() * 1e-8 if s.size and s.max() > 0 else 0)
        U = U[:, keep]
        P = U @ U.T                                # projector on span(A)
        self.M = Winv @ P @ W                      # x -> component to remove
        self.rank = int(keep.sum())
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return (X - (X - self.mu_x) @ self.M.T).astype(np.float32)


def z_readability(Xtr, ztr, Xte, zte, seed: int = 0) -> float:
    """Balanced accuracy of a logistic probe for the categorical z after
    (or before) erasure -- the check that the eraser did its job. Classes
    absent from train are scored as errors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ztr)
    pred = clf.predict(sc.transform(Xte))
    return float(balanced_accuracy_score(zte, pred))
