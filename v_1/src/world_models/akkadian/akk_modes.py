"""Two evaluation modes that address the year==ruler-identity confound, mirroring the
thesis's balanced-MC machinery (shared/mc_maxking.py, shared/mc_probe.py):

  mc_balanced  — in-distribution, ruler frequency balanced out. 200 balanced draws
                 (cap = min ruler count, e.g. 21 for r8); within each draw a
                 StratifiedKFold-by-ruler (rulers appear in train AND test), ridge,
                 out-of-fold scored per draw, averaged over draws (mean ± std). The
                 thesis's `year_strat` analog: "can you read the date, imbalance
                 removed, rulers seen."

  loro         — leave-ONE-ruler-out (GroupKFold-by-ruler). Train on all-but-one
                 ruler, predict the held-out ruler's fragments; pool the out-of-fold
                 predictions over all rulers and score ONCE. The thesis's `year_group`
                 analog and the real generalization test: "can you place an UNSEEN
                 ruler?" Spearman over the pooled predictions is the headline (does it
                 order rulers it never trained on); R² adds calibration.

Both operate on one layer's activations at a time; the caller sweeps layers.
Ridge alpha fixed to n_features (the paper's heuristic) for speed across many fits.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_lib import probing  # noqa: E402


def _ridge_predict(Xtr, ytr, Xte, alpha):
    mu, sd = ytr.mean(axis=0), ytr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    r = Ridge(alpha=alpha).fit(Xtr, (ytr - mu) / sd)
    return r.predict(Xte) * sd + mu


def _score(y_true, y_pred, is_place):
    return (probing.score_place(y_true, y_pred) if is_place
            else probing.score_time(y_true, y_pred))


def mc_balanced(X, y, ruler, is_place, cap=None, n_draws=200, n_splits=5, seed=42,
                alpha=None):
    """Balanced Monte-Carlo at ONE layer. Returns dict with mean/std of R² and
    Spearman over draws, plus the per-draw Spearman series and the cap used."""
    rng = np.random.RandomState(seed)
    alpha = float(X.shape[1]) if alpha is None else float(alpha)
    rulers = np.unique(ruler)
    per_ruler = {r: np.flatnonzero(ruler == r) for r in rulers}
    counts = {r: len(ix) for r, ix in per_ruler.items()}
    if cap is None:
        cap = min(counts.values())
    r2s, sps = [], []
    for d in range(n_draws):
        rows = np.concatenate([
            rng.choice(per_ruler[r], size=min(cap, counts[r]), replace=False)
            for r in rulers])
        Xs, ys, gs = X[rows], y[rows], ruler[rows]
        codes = np.unique(gs, return_inverse=True)[1]
        ns = min(n_splits, int(np.bincount(codes).min()))
        if ns < 2:
            continue
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=d)
        oof = np.full(ys.shape, np.nan)
        for tr, te in skf.split(Xs, codes):
            oof[te] = _ridge_predict(Xs[tr], ys[tr], Xs[te], alpha)
        sc = _score(ys, oof, is_place)
        r2s.append(sc["r2"])
        sps.append(sc.get("spearman",
                          (sc.get("lat_spearman", np.nan)
                           + sc.get("lon_spearman", np.nan)) / 2))
    ok = lambda v: [x for x in v if x == x]  # noqa: E731
    r2ok, spok = ok(r2s), ok(sps)
    return {
        "mode": "mc_balanced", "cap": int(cap), "n_draws_used": len(r2ok),
        "r2_mean": float(np.mean(r2ok)) if r2ok else float("nan"),
        "r2_std": float(np.std(r2ok)) if r2ok else float("nan"),
        "spearman_mean": float(np.mean(spok)) if spok else float("nan"),
        "spearman_std": float(np.std(spok)) if spok else float("nan"),
    }


def mc_site(X, y, site, cap=None, n_draws=200, n_splits=5, seed=42, alpha=None):
    """Balanced Monte-Carlo BY FIND-SPOT (the space analog of mc_balanced, exact
    mirror of its ruler protocol): draws balanced across merged sites (cap per site),
    StratifiedKFold-by-site within each draw so every site appears in train AND test —
    in-distribution, site-imbalance removed, paper-comparable. y is (n,2) [lon,lat];
    scores lon/lat R² + mean lat/lon Spearman. `site` is a per-row merged-site key;
    rows with site None must be filtered by the caller. (For the harder "place an
    UNSEEN find-spot" generalization, group-hold-out is a separate test.)"""
    rng = np.random.RandomState(seed)
    alpha = float(X.shape[1]) if alpha is None else float(alpha)
    sites = np.unique(site)
    per_site = {s: np.flatnonzero(site == s) for s in sites}
    counts = {s: len(ix) for s, ix in per_site.items()}
    if cap is None:
        cap = min(counts.values())
    r2s, sps = [], []
    for d in range(n_draws):
        rows = np.concatenate([
            rng.choice(per_site[s], size=min(cap, counts[s]), replace=False)
            for s in sites])
        Xs, ys, gs = X[rows], y[rows], site[rows]
        codes = np.unique(gs, return_inverse=True)[1]
        ns = min(n_splits, int(np.bincount(codes).min()))
        if ns < 2:
            continue
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=d)
        oof = np.full(ys.shape, np.nan)
        for tr, te in skf.split(Xs, codes):
            oof[te] = _ridge_predict(Xs[tr], ys[tr], Xs[te], alpha)
        sc = _score(ys, oof, True)
        r2s.append(sc["r2"])
        sps.append((sc.get("lat_spearman", np.nan)
                    + sc.get("lon_spearman", np.nan)) / 2)
    ok = lambda v: [x for x in v if x == x]  # noqa: E731
    r2ok, spok = ok(r2s), ok(sps)
    return {
        "mode": "mc_site", "cap": int(cap), "n_sites": int(len(sites)),
        "n_draws_used": len(r2ok),
        "r2_mean": float(np.mean(r2ok)) if r2ok else float("nan"),
        "r2_std": float(np.std(r2ok)) if r2ok else float("nan"),
        "spearman_mean": float(np.mean(spok)) if spok else float("nan"),
        "spearman_std": float(np.std(spok)) if spok else float("nan"),
    }


def loro(X, y, ruler, is_place, alpha=None):
    """Leave-one-ruler-out at ONE layer. Pools out-of-fold predictions over all
    rulers and scores once. Returns R² + Spearman on the pooled predictions."""
    alpha = float(X.shape[1]) if alpha is None else float(alpha)
    rulers = np.unique(ruler)
    if len(rulers) < 3:
        return {"mode": "loro", "skipped": True, "n_rulers": int(len(rulers))}
    oof = np.full(y.shape, np.nan)
    for r in rulers:
        te = ruler == r
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        oof[te] = _ridge_predict(X[tr], y[tr], X[te], alpha)
    valid = np.isfinite(oof) if oof.ndim == 1 else np.isfinite(oof).all(axis=1)
    sc = _score(y[valid], oof[valid], is_place)
    sp = sc.get("spearman",
                (sc.get("lat_spearman", np.nan) + sc.get("lon_spearman", np.nan)) / 2)
    return {"mode": "loro", "n_rulers": int(len(rulers)),
            "n_scored": int(valid.sum()),
            "r2": float(sc["r2"]), "spearman": float(sp)}
