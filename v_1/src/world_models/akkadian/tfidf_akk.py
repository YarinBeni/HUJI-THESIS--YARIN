"""WA TF-IDF floor for the Akkadian mimic: bag-of-substrings of the fragment text
(Akkadian or English) -> year / coords. The floor every embedding arm must clear.

    python tfidf_akk.py            # both variants x r8/r40 x year/geo
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import akk_data as A               # noqa: E402
import akk_modes as M              # noqa: E402
from wm_lib import probing         # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
ALPHAS = np.logspace(-2, 4, 13)
SEED = 42


def run(df, variant, ruler_set, target):
    mask = A.ruler_set_mask(df, ruler_set)
    tgt, valid = A.target_values(df, target)
    sel = mask & valid
    is_place = (target == "geo")
    texts = np.array(A.entity_texts(df, variant), dtype=object)[sel]
    y = tgt[sel]
    is_test = A.is_test_split(df, sel)[sel]
    tr = ~is_test

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                           lowercase=True, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_features=120000, lowercase=True, sublinear_tf=True)
    Xw = word.fit(texts[tr]).transform(texts)
    Xc = char.fit(texts[tr]).transform(texts)
    X = hstack([Xw, Xc]).tocsr()

    mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
    yn = (y - mu) / sd
    rng = np.random.RandomState(SEED)
    tri = np.flatnonzero(tr)
    val = rng.choice(tri, size=max(1, len(tri) // 10), replace=False)
    fit = np.setdiff1d(tri, val)
    best = (None, -np.inf)
    for a in ALPHAS:
        r = Ridge(alpha=a, solver="sparse_cg").fit(X[fit], yn[fit])
        v = -np.mean((r.predict(X[val]) - yn[val]) ** 2)
        if v > best[1]:
            best = (a, v)
    ridge = Ridge(alpha=best[0], solver="sparse_cg").fit(X[tri], yn[tri])
    proj = ridge.predict(X) * sd + mu

    sfn = probing.score_place if is_place else probing.score_time
    sc = {"train": sfn(y[tr], proj[tr]), "test": sfn(y[~tr], proj[~tr])}
    bsp = sc["test"].get("spearman",
                         (sc["test"].get("lat_spearman", float("nan"))
                          + sc["test"].get("lon_spearman", float("nan"))) / 2)

    # balanced-MC + leave-one-ruler-out on the same TF-IDF features (alpha = the
    # holdout-selected value, since n_features ~1e5 makes the paper heuristic huge)
    ruler = df.ruler.values[sel]
    mc = M.mc_balanced(X, y, ruler, is_place, n_draws=200, alpha=best[0])
    lo = M.loro(X, y, ruler, is_place, alpha=best[0])

    out = {"method": "tfidf", "variant": variant, "ruler_set": ruler_set,
           "target": target, "site": "text", "n": int(sel.sum()),
           "n_rulers": int(len(np.unique(ruler))), "n_features": int(X.shape[1]),
           "holdout": {"best_layer": 0, "best_test_r2": sc["test"]["r2"],
                       "best_test_spearman": float(bsp), "n_test": int(is_test.sum())},
           "mc": {**mc, "layer": 0}, "loro": lo,
           "best_layer": 0, "best_test_r2": sc["test"]["r2"],
           "best_test_spearman": float(bsp)}
    pdir = os.path.join(RESULTS_DIR, "probes", "tfidf")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{variant}.{ruler_set}.{target}.text.ridge.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tfidf/{variant}/{ruler_set}/{target}] holdout r2={sc['test']['r2']:.3f}"
          f" | mc rho={mc['spearman_mean']:.3f} | loro rho={lo.get('spearman',float('nan')):.3f}",
          flush=True)


if __name__ == "__main__":
    df = A.load_fragments()
    for variant in A.TEXT_VARIANTS:
        for ruler_set in A.RULER_SETS:
            for target in A.TARGETS:
                run(df, variant, ruler_set, target)
