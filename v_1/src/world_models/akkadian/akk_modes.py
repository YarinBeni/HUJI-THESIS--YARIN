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


#: PLS latent-dimension grid. The thesis deck swept only {1,2,3,5}
#: (stress_tests/shared/mc_probe.py PLS_KS) and 18 of our 58 fragment cells came back
#: pinned at that ceiling — i.e. the grid, not the data, was choosing k. Extended
#: log-spaced to 64; the upper end is guarded per-fold by `k < min(len(train), D)`.
PLS_KS = (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64)


def mc_group(X, y, ruler, cap=None, n_draws=200, n_splits=5, seed=42, alpha=None,
             pls_ks=PLS_KS, nested_every=4):
    """Balanced Monte-Carlo with GroupKFold-BY-RULER — the thesis deck's protocol.

    Mirrors stress_tests/shared/mc_probe.py, the engine behind the headline table
    (p1_year_mc.csv; deck slide 2: "200 Monte-Carlo draws over 8 balanced rulers,
    GroupKFold by ruler. Closes: ... ruler leakage").

    The difference from `mc_balanced` is the only thing that matters here: a ruler is
    wholly in train OR wholly in test, so the probe cannot re-identify a ruler's scribal
    style and read the year off it. Since r8 `year` has just 17 distinct values across 8
    rulers, that leak is worth ~0.4 Spearman for a pure n-gram baseline.

    Per mc_probe: rows are L2-normalised inside each draw; each fold is scored separately
    and the folds are averaged (NaN folds — a test fold holding a single ruler has
    constant y, so Spearman is undefined — are dropped); draws are then averaged.
    Reports ridge plus a PLS sweep over `pls_ks`.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.RandomState(seed)
    alpha = float(X.shape[1]) if alpha is None else float(alpha)
    rulers = np.unique(ruler)
    per_ruler = {r: np.flatnonzero(ruler == r) for r in rulers}
    counts = {r: len(ix) for r, ix in per_ruler.items()}
    if cap is None:
        cap = min(counts.values())

    def _l2(A):
        n = np.linalg.norm(A, axis=1, keepdims=True)
        return A / np.where(n == 0, 1.0, n)

    ridge_sp, ridge_r2 = [], []
    pls_sp = {k: [] for k in pls_ks}
    nested_sp, nested_ks = [], []
    for d in range(n_draws):
        rows = np.concatenate([
            rng.choice(per_ruler[r], size=min(cap, counts[r]), replace=False)
            for r in rulers])
        Xs, ys, gs = _l2(X[rows].astype(np.float64)), y[rows], ruler[rows]
        nr = len(np.unique(gs))
        if len(Xs) < 10 or nr < 2:
            continue
        ns = min(n_splits, nr)
        fold_sp, fold_r2 = [], []
        fold_pls = {k: [] for k in pls_ks}
        for tr, te in GroupKFold(n_splits=ns).split(Xs, ys, groups=gs):
            if len(np.unique(ys[te])) < 2:      # single-ruler fold -> rho undefined
                continue
            pred = _ridge_predict(Xs[tr], ys[tr], Xs[te], alpha)
            sc = _score(ys[te], pred, False)
            fold_sp.append(sc["spearman"]); fold_r2.append(sc["r2"])
            for k in pls_ks:
                if k >= min(len(tr), Xs.shape[1]):
                    continue
                try:
                    p = PLSRegression(n_components=k).fit(Xs[tr], ys[tr].reshape(-1, 1))
                    fold_pls[k].append(_score(ys[te], p.predict(Xs[te]).ravel(),
                                              False)["spearman"])
                except Exception:                                    # noqa: BLE001
                    pass
            # Nested k: pick k on an inner GroupKFold over the TRAINING rulers only,
            # then refit on the full train fold. `pls_best_k` below is selected on the
            # outer test folds (that is what the deck does, so it stays for
            # comparability) — but that biases the headline upward, and the bias grows
            # with the size of the grid. This is the unbiased counterpart.
            if d % nested_every == 0:
                gtr = gs[tr]
                if len(np.unique(gtr)) >= 3:
                    inner = {}
                    for itr, ite in GroupKFold(
                            n_splits=min(3, len(np.unique(gtr)))).split(
                            Xs[tr], ys[tr], groups=gtr):
                        if len(np.unique(ys[tr][ite])) < 2:
                            continue
                        for k in pls_ks:
                            if k >= min(len(itr), Xs.shape[1]):
                                continue
                            try:
                                p = PLSRegression(n_components=k).fit(
                                    Xs[tr][itr], ys[tr][itr].reshape(-1, 1))
                                s = _score(ys[tr][ite],
                                           p.predict(Xs[tr][ite]).ravel(),
                                           False)["spearman"]
                                if s == s:
                                    inner.setdefault(k, []).append(s)
                            except Exception:                        # noqa: BLE001
                                pass
                    if inner:
                        kbest = max(inner, key=lambda k: np.mean(inner[k]))
                        try:
                            p = PLSRegression(n_components=kbest).fit(
                                Xs[tr], ys[tr].reshape(-1, 1))
                            s = _score(ys[te], p.predict(Xs[te]).ravel(),
                                       False)["spearman"]
                            if s == s:
                                nested_sp.append(float(s))
                                nested_ks.append(int(kbest))
                        except Exception:                            # noqa: BLE001
                            pass
        ok = lambda v: [x for x in v if x == x]                      # noqa: E731
        if ok(fold_sp):
            ridge_sp.append(float(np.mean(ok(fold_sp))))
            ridge_r2.append(float(np.mean(ok(fold_r2))))
        for k in pls_ks:
            if ok(fold_pls[k]):
                pls_sp[k].append(float(np.mean(ok(fold_pls[k]))))

    def agg(v):
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
    sp_m, sp_s = agg(ridge_sp)
    r2_m, r2_s = agg(ridge_r2)
    per_k = {str(k): {"spearman_mean": agg(pls_sp[k])[0],
                      "spearman_std": agg(pls_sp[k])[1]} for k in pls_ks}
    best_k = max((k for k in pls_ks if pls_sp[k]),
                 key=lambda k: np.mean(pls_sp[k]), default=None)
    return {
        "mode": "mc_group", "splitter": "GroupKFold-by-ruler", "cap": int(cap),
        "n_draws_used": len(ridge_sp),
        "spearman_mean": sp_m, "spearman_std": sp_s,
        "r2_mean": r2_m, "r2_std": r2_s,
        "pls_per_k": per_k, "pls_best_k": (int(best_k) if best_k else None),
        "pls_spearman_mean": (agg(pls_sp[best_k])[0] if best_k else float("nan")),
        "pls_ks": list(pls_ks),
        "pls_k_at_grid_ceiling": bool(best_k == max(pls_ks)),
        # nested = k chosen inside the training rulers, so not selected on test
        "pls_nested_spearman_mean": agg(nested_sp)[0],
        "pls_nested_spearman_std": agg(nested_sp)[1],
        "pls_nested_k_median": (float(np.median(nested_ks)) if nested_ks else float("nan")),
        "pls_nested_n_folds": len(nested_sp),
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
