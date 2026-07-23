"""English (G&T redo) best-layer PLS-k sweep — the English analog of the Akkadian
layers_pls job, so the encoders (AKK-300M, cuneiform-400M, uMT5) enter the PLS plots.

For one method, per entity dataset and pooling site on disk: pick the best layer by
ridge test-R², then at that layer sweep PLS k in {1,2,3,5,8,16,32,64} and record
R²+Spearman per k. Reuses the stored English acts (CPU, no re-extraction).

Writes results/eng_pls/{method}/{entity_type}.{site}.json
    python probe_eng_pls.py --method thalesian_cunei400m
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
from wm_lib import entity_data                 # noqa: E402
from wm_lib import probing                     # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
KS = [1, 2, 3, 5, 8, 16, 32, 64]


def _sp(sc):
    return sc.get("spearman", (sc.get("lat_spearman", np.nan)
                               + sc.get("lon_spearman", np.nan)) / 2)


def probe_one(method, entity_type, site):
    act_dir = os.path.join(ACTS_DIR, method, entity_type)
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    if not files:
        return None
    df = entity_data.load_entity_df(entity_type)
    target, valid = entity_data.target_values(entity_type, df)
    is_test = df.is_test.values.astype(bool)
    _, is_place = entity_data.FEATURES[entity_type]
    meta = json.load(open(os.path.join(act_dir, "metadata.json")))
    n = meta["n_rows"]
    target, valid, is_test = target[:n], valid[:n], is_test[:n]

    layers, per_layer, best = {}, [], (None, -np.inf)
    for p in files:
        li = int(re.search(r"layer(\d+)\.npz$", p).group(1))
        X = np.load(p)["acts"][:n][valid]
        if np.isnan(X).any():
            continue
        layers[li] = X
        sc, _, _ = probing.run_probe(X, target[valid], is_test[valid], is_place)
        per_layer.append({"layer": li, "test_r2": float(sc["test"]["r2"]),
                          "test_spearman": float(_sp(sc["test"]))})
        if sc["test"]["r2"] > best[1]:
            best = (li, sc["test"]["r2"])
    if not layers:
        return None
    lids = sorted(layers); bl = best[0]
    for r in per_layer:
        r["nd"] = round((r["layer"] - lids[0]) / max(1, lids[-1] - lids[0]), 4)
    pls = {}
    for k in KS:
        try:
            sc, _, _ = probing.run_pls_probe(layers[bl], target[valid],
                                             is_test[valid], is_place, k=k)
            pls[str(k)] = {"test_r2": float(sc["test"]["r2"]),
                           "test_spearman": float(_sp(sc["test"]))}
        except Exception as e:                                    # noqa: BLE001
            pls[str(k)] = {"error": str(e)[:80]}
    bk = max((k for k in pls if "test_r2" in pls[k]),
             key=lambda k: pls[k]["test_r2"], default=None)
    out = {"method": method, "entity_type": entity_type, "site": site,
           "is_place": bool(is_place), "best_layer": bl,
           "best_k": int(bk) if bk else None,
           "per_layer": per_layer, "pls_at_best_layer": pls}
    pdir = os.path.join(RESULTS_DIR, "eng_pls", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{entity_type}.{site}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{method}/{entity_type}/{site}] best L{bl} r2={best[1]:.3f} best_k={bk}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    args = ap.parse_args()
    mdir = os.path.join(ACTS_DIR, args.method)
    for et in entity_data.ENTITY_TYPES:
        act_dir = os.path.join(mdir, et)
        if not os.path.isdir(act_dir):
            continue
        sites = {re.match(r"(\w+)\.layer", os.path.basename(p)).group(1)
                 for p in glob.glob(os.path.join(act_dir, "*.layer*.npz"))}
        for site in sorted(sites):
            probe_one(args.method, et, site)


if __name__ == "__main__":
    main()
