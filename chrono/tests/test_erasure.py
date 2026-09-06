import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.eval.erasure import LeaceEraser, z_readability  # noqa: E402


def _data(n=600, d=30, k=4, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(k, size=n)
    Z = np.eye(k)[z]
    X = rng.normal(size=(n, d)) + Z @ rng.normal(scale=0.6, size=(k, d))   # class shift small vs noise
    return X, Z, z


def test_erased_features_have_zero_cross_covariance_with_z():
    X, Z, z = _data()
    er = LeaceEraser().fit(X, Z)
    Xe = er(X)
    cross = (Xe - Xe.mean(0)).T @ (Z - Z.mean(0)) / len(X)
    assert np.abs(cross).max() < 1e-6, np.abs(cross).max()
    assert er.rank == 3                       # k-1 independent directions


def test_erasure_kills_linear_readability_but_is_minimal():
    X, Z, z = _data()
    tr, te = np.arange(400), np.arange(400, 600)
    er = LeaceEraser().fit(X[tr], Z[tr])
    before = z_readability(X[tr], z[tr], X[te], z[te])
    after = z_readability(er(X[tr]), z[tr], er(X[te]), z[te])
    assert before > 0.9 and after < 0.4, (before, after)   # chance = .25
    # minimality: the change is confined to <= k-1 directions, so most
    # variance survives
    kept = np.var(er(X), axis=0).sum() / np.var(X, axis=0).sum()
    assert kept > 0.6, kept


def test_mutation_no_erasure_would_fail():
    """The readability assertion above is not vacuous: identity 'eraser'."""
    X, Z, z = _data()
    tr, te = np.arange(400), np.arange(400, 600)
    assert z_readability(X[tr], z[tr], X[te], z[te]) > 0.9
