"""Add the deck's GroupKFold-by-ruler protocol to the fragment probe results.

The world-models fragment JSONs currently carry `holdout`, `mc` (StratifiedKFold-by-ruler
— rulers appear in train AND test) and `loro`. The thesis deck's headline protocol is
neither: it is GroupKFold-by-ruler inside the 200 balanced draws
(stress_tests/shared/mc_probe.py -> p1_year_mc.csv -> slide 4). Because r8 `year` is
essentially a ruler label, the stratified variant leaks ruler identity and inflates every
arm — most visibly TF-IDF, which reaches .707 on name-stripped text where the deck reports
.266.

This script computes `mc_group` at the holdout-best layer and MERGES it into the existing
JSON, leaving every other key untouched, so both protocols stay available and no earlier
figure silently changes.

    python probe_akk_group.py --method qwen3_8b
    python probe_akk_group.py --tfidf
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import akk_data as A                          # noqa: E402
import akk_modes as M                         # noqa: E402
from wm_lib import probing                    # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
SITES = ["last", "mean"]
VARIANTS = ["akk_maximal", "eng_tier0"]
MC_DRAWS = 200


def _merge(path, block):
    d = json.load(open(path))
    d["mc_group"] = block
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def one(method, variant, site, df, args):
    path = os.path.join(RESULTS_DIR, "probes", method,
                        f"{variant}.r8.year.{site}.ridge.json")
    if not os.path.exists(path):
        return None
    prev = json.load(open(path))
    if "mc_group" in prev and not args.force:
        print(f"[skip] {method}/{variant}/{site} already has mc_group", flush=True)
        return None
    mask = A.ruler_set_mask(df, "r8")
    tgt, valid = A.target_values(df, "year")
    sel = mask & valid & (df.year.values >= 100)      # 6 rulers carry ordinal junk years
    y = tgt[sel]
    ruler = df.ruler.values[sel]

    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        texts = np.array(A.entity_texts(df, variant), dtype=object)[sel]
        V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=120000, sublinear_tf=True)
        X = TruncatedSVD(n_components=256, random_state=42).fit_transform(
            V.fit_transform(texts)).astype(np.float32)
    else:
        bl = prev.get("holdout", {}).get("best_layer", prev.get("best_layer"))
        fp = os.path.join(ACTS_DIR, method, variant, f"{site}.layer{bl}.npz")
        if not os.path.exists(fp):
            print(f"[skip] {method}/{variant}/{site}: no acts for layer {bl}", flush=True)
            return None
        X, bad = probing.sanitize(np.load(fp)["acts"][sel].astype(np.float32))
        if bad > 0.01:
            print(f"[warn] {method}/{variant}/{site}: {bad:.1%} non-finite", flush=True)

    blk = M.mc_group(X, y, ruler, n_draws=args.n_draws)
    blk["layer"] = 0 if method == "tfidf" else int(
        prev.get("holdout", {}).get("best_layer", 0))
    _merge(path, blk)
    print(f"[{method}/{variant}/{site}] mc_group rho={blk['spearman_mean']:.3f}"
          f"±{blk['spearman_std']:.3f} | PLS k={blk['pls_best_k']} "
          f"rho={blk['pls_spearman_mean']:.3f}  (mc was "
          f"{prev.get('mc',{}).get('spearman_mean',float('nan')):.3f})", flush=True)
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--n-draws", type=int, default=MC_DRAWS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    df = A.load_fragments()
    method = "tfidf" if args.tfidf else args.method
    if not method:
        ap.error("give --method or --tfidf")
    sites = ["text"] if method == "tfidf" else SITES
    for variant in VARIANTS:
        for site in sites:
            try:
                one(method, variant, site, df, args)
            except Exception as e:                                  # noqa: BLE001
                print(f"[error] {method}/{variant}/{site}: {type(e).__name__}: {e}",
                      flush=True)


if __name__ == "__main__":
    main()
