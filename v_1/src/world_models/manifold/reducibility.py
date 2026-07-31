"""Irreducibility diagnostics (Engels et al., "Not All Language Model Features Are Linear").

Their question is not "is there a direction" but "is this 2-D cloud REALLY 2-D, or does it
factor into something simpler?". Two indices, computed on a pair of PCA coordinates:

  * epsilon-mixture index M_eps  — is there a direction with a GAP, i.e. is the cloud a
    mixture of separated linear pieces? Fraction of points inside the normalised
    eps-slab of the best separating direction. LOW  M => reducible (a clean gap exists).
  * separability index S         — can a rotation make the two coordinates statistically
    independent? Minimum mutual information (bits) over rotations. LOW S => reducible
    (two independent 1-D features). A circle scores HIGH on both => irreducibly 2-D.

Their years-of-the-20th-century manifold is the reference "irreducible ring". We reuse the
indices to ask whether our year / geo representations are rings, lines, or blobs.

Calibrate with `synthetic_baselines()` (gaussian / two-gaussians / circle / lattice) —
their `reducibility_demo.py` equivalent — before reading anything into real numbers.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ mixture index
def mixture_index(xy, eps=0.1, iters=2000, lr=0.1, seed=0):
    """M_eps: fraction of points inside the normalised eps-slab of the best direction.

    We search the direction analytically over a dense angle grid (their version does
    gradient ascent on a sigmoid surrogate; the 1-D search is equivalent and has no
    optimisation knobs). Returns (M, best_angle)."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return float("nan"), float("nan")
    ths = np.linspace(0, np.pi, 361)[:-1]
    best = (np.inf, np.nan)
    for th in ths:
        a = np.array([np.cos(th), np.sin(th)])
        p = xy @ a
        p = p - p.mean()
        s = np.sqrt(np.mean(p ** 2))
        if s == 0:
            continue
        z = p / s
        M = float(np.mean(np.abs(z) < eps))
        if M < best[0]:
            best = (M, float(th))
    return best


# --------------------------------------------------------------- separability index
def _mutual_information(xy, bins=10):
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    J = H / max(H.sum(), 1)
    px = J.sum(axis=1, keepdims=True)
    py = J.sum(axis=0, keepdims=True)
    denom = px @ py
    m = (J > 0) & (denom > 0)
    return float(np.sum(J[m] * np.log(J[m] / denom[m])) / np.log(2))  # bits


def separability_index(xy, n_angles=100, bins=10):
    """S: minimum mutual information (bits) over rotations. LOW => separable."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return float("nan"), float("nan")
    xy = xy - xy.mean(0)
    best = (np.inf, np.nan)
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        mi = _mutual_information(xy @ R.T, bins=bins)
        if mi < best[0]:
            best = (mi, float(th))
    return best


def punch_out(xy, radius=0.0):
    """Drop the dense blob at the origin (their --radius knob); it otherwise dominates
    both indices."""
    if radius <= 0:
        return np.ones(len(xy), dtype=bool)
    return np.linalg.norm(np.asarray(xy, dtype=float), axis=1) > radius


def pair_indices(P, pairs=((0, 1), (1, 2), (2, 3), (3, 4)), eps=0.1, radius=0.0):
    """Both indices for consecutive PC pairs. Their years result lived in PCs 3-4, so
    never look at PC1-2 alone."""
    out = {}
    for i, j in pairs:
        if max(i, j) >= P.shape[1]:
            continue
        xy = P[:, [i, j]]
        m = punch_out(xy, radius)
        if m.sum() < 10:
            continue
        M, _ = mixture_index(xy[m], eps=eps)
        S, _ = separability_index(xy[m])
        r = np.linalg.norm(xy[m] - xy[m].mean(0), axis=1)
        out[f"{i}-{j}"] = {
            "mixture_index": M, "separability_index": S,
            "score": (1 - M) * S if np.isfinite(M) and np.isfinite(S) else float("nan"),
            "radius_mean": float(r.mean()), "radius_cv": float(r.std() / max(r.mean(), 1e-9)),
            "n": int(m.sum()),
        }
    return out


def centered_spectrum(X, k=12):
    """Singular values of X and of the mean-CENTERED X. Their circle diagnostic: after
    centering, two comparable leading singular values => a ring, not a spike."""
    X = np.asarray(X, dtype=float)
    s_raw = np.linalg.svd(X, compute_uv=False)[:k]
    s_cen = np.linalg.svd(X - X.mean(0), compute_uv=False)[:k]
    return s_raw.tolist(), s_cen.tolist()


# ------------------------------------------------------------------- calibration
def synthetic_baselines(n=1000, seed=0):
    """Reference values so the real numbers are readable (their reducibility_demo)."""
    rng = np.random.RandomState(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    sets = {
        "gaussian": rng.randn(n, 2),
        "two_gaussians": np.vstack([rng.randn(n // 2, 2) + [3, 0],
                                    rng.randn(n // 2, 2) - [3, 0]]),
        "circle": np.c_[np.cos(th), np.sin(th)] + 0.05 * rng.randn(n, 2),
        "lattice": rng.randint(0, 5, (n, 2)) + 0.05 * rng.randn(n, 2),
    }
    out = {}
    for k, xy in sets.items():
        M, _ = mixture_index(xy)
        S, _ = separability_index(xy)
        out[k] = {"mixture_index": M, "separability_index": S}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_baselines(), indent=2))
