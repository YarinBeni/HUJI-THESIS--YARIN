"""By-SITE geo probe (the paper's space protocol, reported in R²).

For one method, over the EXISTING Akkadian geo activations (no re-extraction), for
each text variant on disk and each pooling site (last, mean): pick the best layer by a
within-split holdout, then run balanced Monte-Carlo BY FIND-SPOT at that layer
(10 merged sites, cap 21, 200 draws, GroupKFold-by-site) and report the paper's R² +
Spearman on (lon, lat). This is the by-site replacement for the by-ruler geo MC.

Writes results/probes_geosite/{method}/{variant}.{pool}.geo_site.json

    python probe_geo_site.py --method qwen3_8b
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
MC_DRAWS = 200
MIN_COUNT = 18


def _load_layers(act_dir, site, sel):
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    layers = {}
    for path in files:
        li = int(re.search(r"layer(\d+)\.npz$", path).group(1))
        X = np.load(path)["acts"][sel].astype(np.float32)
        X, bad = probing.sanitize(X)
        if bad <= 0.01:
            layers[li] = X
        else:
            print(f"[warn] layer {li}: {bad:.1%} non-finite, skipped", flush=True)
    return layers


def _tfidf_features(df, variant, sel, is_test):
    """char_wb(2,5)+word(1,2) TF-IDF -> SVD-256, fit on the holdout-train rows only."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    texts = np.array(A.entity_texts(df, variant), dtype=object)[sel]
    tr = ~is_test
    ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                         max_features=120000, lowercase=True, sublinear_tf=True)
    wd = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                         lowercase=True, sublinear_tf=True)
    X = hstack([ch.fit(texts[tr]).transform(texts), wd.fit(texts[tr]).transform(texts)])
    return TruncatedSVD(n_components=256, random_state=42).fit_transform(X)


def probe_one(method, variant, pool, df, site_lab):
    act_dir = os.path.join(ACTS_DIR, method, variant)
    y_all, valid = A.target_values(df, "geo")          # (n,2) lon/lat, has_geo mask
    sel = valid & np.array([s is not None for s in site_lab])
    if sel.sum() < 20:
        return None
    y = y_all[sel]
    site = np.array([s for s in site_lab[sel]])
    is_test = A.is_test_split(df, sel)[sel]

    if method == "tfidf":
        Xf = _tfidf_features(df, variant, sel, is_test)
        layers = {0: Xf}
    else:
        layers = _load_layers(act_dir, pool, sel)
    if not layers:
        return None

    # best layer by a held-out-by-ruler split (cheap, r2)
    best = (None, -np.inf)
    for li, X in layers.items():
        scores, _, _ = probing.run_probe(X, y, is_test, True)
        if scores["test"]["r2"] > best[1]:
            best = (li, scores["test"]["r2"])
    bl = best[0]

    ms = M.mc_site(layers[bl], y, site, cap=None, n_draws=MC_DRAWS)
    ms["layer"] = bl
    out = {"method": method, "variant": variant, "pool": pool, "target": "geo",
           "protocol": "by_site", "n": int(sel.sum()),
           "holdout": {"best_layer": bl, "best_test_r2": float(best[1])},
           "mc_site": ms}
    pdir = os.path.join(RESULTS_DIR, "probes_geosite", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{variant}.{pool}.geo_site.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{method}/{variant}/{pool}] by-site geo: layer {bl} | "
          f"R2={ms['r2_mean']:.3f}±{ms['r2_std']:.3f} | rho={ms['spearman_mean']:.3f} "
          f"(cap={ms['cap']}, {ms['n_sites']} sites)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    df = A.load_fragments()
    site_lab = A.merged_site_labels(df, min_count=MIN_COUNT)
    if args.method == "tfidf":
        variants = [v for v in A.TEXT_VARIANTS
                    if args.variant is None or v == args.variant]
        pools = ["text"]
    else:
        variants = [v for v in A.TEXT_VARIANTS
                    if os.path.isdir(os.path.join(ACTS_DIR, args.method, v))
                    and (args.variant is None or v == args.variant)]
        pools = SITES
    for variant in variants:
        for pool in pools:
            probe_one(args.method, variant, pool, df, site_lab)


if __name__ == "__main__":
    main()
