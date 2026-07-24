"""Per-layer R² curve + best-layer PLS-k sweep for the Akkadian probes.

For one method, over the stored acts (eng_tier0 + akk_maximal, year + geo, last + mean):
  * per layer: held-out ridge R² + Spearman (for the layer-depth curve, normalized);
  * at the best layer: PLS with k in {1,2,3,5,8,16,32,64}, R²+Spearman per k (best-k).

Writes results/layers_pls/{method}/{variant}.{target}.{site}.json  (CPU, reuses acts).
    python probe_layers_pls.py --method qwen3_8b
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
from wm_lib import probing                    # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
SITES = ["last", "mean"]
KS = [1, 2, 3, 5, 8, 16, 32, 64]


def _layers(act_dir, site, sel):
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    out = {}
    for p in files:
        li = int(re.search(r"layer(\d+)\.npz$", p).group(1))
        X = np.load(p)["acts"][sel].astype(np.float32)
        X, bad = probing.sanitize(X)
        if bad <= 0.01:
            out[li] = X
        else:
            print(f"[warn] layer {li}: {bad:.1%} non-finite, skipped", flush=True)
    return out


def _sp(sc):
    return sc.get("spearman", (sc.get("lat_spearman", np.nan)
                               + sc.get("lon_spearman", np.nan)) / 2)


def probe_one(method, variant, target, site, df):
    act_dir = os.path.join(ACTS_DIR, method, variant)
    y, valid = A.target_values(df, target)
    sel = valid
    y = y[sel]
    is_place = (target == "geo")
    is_test = A.is_test_split(df, sel)[sel]
    layers = _layers(act_dir, site, sel)
    if not layers:
        return None
    lids = sorted(layers)
    per_layer, best = [], (None, -np.inf)
    for li in lids:
        sc, _, _ = probing.run_probe(layers[li], y, is_test, is_place)
        nd = (li - lids[0]) / max(1, (lids[-1] - lids[0]))
        per_layer.append({"layer": li, "nd": round(nd, 4),
                          "test_r2": float(sc["test"]["r2"]),
                          "test_spearman": float(_sp(sc["test"]))})
        if sc["test"]["r2"] > best[1]:
            best = (li, sc["test"]["r2"])
    bl = best[0]
    pls = {}
    for k in KS:
        try:
            sc, _, _ = probing.run_pls_probe(layers[bl], y, is_test, is_place, k=k)
            pls[str(k)] = {"test_r2": float(sc["test"]["r2"]),
                           "test_spearman": float(_sp(sc["test"]))}
        except Exception as e:                                     # noqa: BLE001
            pls[str(k)] = {"error": str(e)[:80]}
    bk = max((k for k in pls if "test_r2" in pls[k]),
             key=lambda k: pls[k]["test_r2"], default=None)
    out = {"method": method, "variant": variant, "target": target, "site": site,
           "n": int(sel.sum()), "best_layer": bl, "best_k": int(bk) if bk else None,
           "per_layer": per_layer, "pls_at_best_layer": pls}
    pdir = os.path.join(RESULTS_DIR, "layers_pls", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{variant}.{target}.{site}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{method}/{variant}/{target}/{site}] best L{bl} r2={best[1]:.3f} | "
          f"best_k={bk} ({len(lids)} layers)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    args = ap.parse_args()
    df = A.load_fragments()
    variants = [v for v in ("akk_maximal", "eng_tier0")
                if os.path.isdir(os.path.join(ACTS_DIR, args.method, v))]
    for variant in variants:
        for target in ("year", "geo"):
            for site in SITES:
                probe_one(args.method, variant, target, site, df)


if __name__ == "__main__":
    main()
