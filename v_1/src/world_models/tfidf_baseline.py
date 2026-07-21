"""TF-IDF surface-form baseline for the W section.

Same split / targets / metrics as the activation probes, but the features are
word 1-2 grams + char_wb 2-5 grams of the exact entity strings the models see.
This is the floor every embedding arm must clear: anything a bag of substrings can
recover ("...ville" is probably in the US, "Ave" is in NYC, "NYT-style words" drift
over decades) is not evidence of a learned world model.

RidgeCV's GCV needs dense X, so alpha is chosen on a 10% carve-out of train
(seed 42) with sparse_cg Ridge, then refit on the full train split.

    python tfidf_baseline.py                # all 6 datasets
    python tfidf_baseline.py --entity-type art --limit 2000
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_lib import entity_data, probing  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
ALPHAS = np.logspace(-2, 4, 13)
SEED = 42


def run_one(entity_type, args):
    df = entity_data.load_entity_df(entity_type)
    strings = entity_data.entity_strings(entity_type, df)
    if args.limit:
        df, strings = df.iloc[:args.limit], strings[:args.limit]
    target, valid = entity_data.target_values(entity_type, df)
    is_test = df.is_test.values.astype(bool)
    feature, is_place = entity_data.FEATURES[entity_type]

    strings = [s for s, v in zip(strings, valid) if v]
    target, is_test = target[valid], is_test[valid]

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                           lowercase=True, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_features=200_000, lowercase=True, sublinear_tf=True)
    tr = ~is_test
    Xw = word.fit(np.array(strings, dtype=object)[tr]).transform(strings)
    Xc = char.fit(np.array(strings, dtype=object)[tr]).transform(strings)
    X = hstack([Xw, Xc]).tocsr()

    y = target
    mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
    y_norm = (y - mu) / sd

    # alpha selection on a train carve-out (GCV unavailable for sparse X)
    rng = np.random.RandomState(SEED)
    tr_idx = np.flatnonzero(tr)
    val_idx = rng.choice(tr_idx, size=max(1, len(tr_idx) // 10), replace=False)
    fit_idx = np.setdiff1d(tr_idx, val_idx)
    best = (None, -np.inf)
    for a in ALPHAS:
        r = Ridge(alpha=a, solver="sparse_cg")
        r.fit(X[fit_idx], y_norm[fit_idx])
        v = 1 - ((r.predict(X[val_idx]) - y_norm[val_idx]) ** 2).sum() / \
            max(((y_norm[val_idx] - y_norm[fit_idx].mean(axis=0)) ** 2).sum(), 1e-9)
        if v > best[1]:
            best = (a, v)
    alpha = best[0]

    ridge = Ridge(alpha=alpha, solver="sparse_cg")
    ridge.fit(X[tr_idx], y_norm[tr_idx])
    proj = ridge.predict(X) * sd + mu

    score_fn = probing.score_place if is_place else probing.score_time
    scores = {"train": score_fn(y[tr], proj[tr]),
              "test": score_fn(y[~tr], proj[~tr]),
              "alpha": float(alpha)}
    out = {
        "method": "tfidf",
        "entity_type": entity_type,
        "feature": feature,
        "site": "text",
        "probe": "ridge",
        "n": int(len(y)),
        "n_test": int((~tr).sum()),
        "n_features": int(X.shape[1]),
        "layers": {"0": scores},
        "best_layer": 0,
        "best_test_r2": scores["test"]["r2"],
        "best_test_spearman": scores["test"].get(
            "spearman", scores["test"].get("lat_spearman")),
    }
    pdir = os.path.join(RESULTS_DIR, "probes", "tfidf")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{entity_type}.text.ridge.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tfidf/{entity_type}] test r2={scores['test']['r2']:.3f} "
          f"(alpha={alpha:g}, {X.shape[1]} feats)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity-type", default="all",
                    choices=["all"] + entity_data.ENTITY_TYPES)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ets = entity_data.ENTITY_TYPES if args.entity_type == "all" else [args.entity_type]
    for et in ets:
        run_one(et, args)


if __name__ == "__main__":
    main()
