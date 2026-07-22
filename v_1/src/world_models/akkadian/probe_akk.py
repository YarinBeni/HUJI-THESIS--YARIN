"""WA probing: ridge probes over the Akkadian activations for one method.

For each text variant on disk, and each (ruler set × target), fit the paper's
per-layer ridge probe on the held-out-by-ruler split. Writes
results/probes/{method}/{variant}.{ruler_set}.{target}.{site}.ridge.json (committed).

    python probe_akk.py --method qwen3_8b
    python probe_akk.py --method llama2_70b --cleanup
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
from wm_lib.registry import MODELS            # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
SITES = ["last", "mean"]


def probe_one(method, variant, ruler_set, target, site, df, args):
    act_dir = os.path.join(ACTS_DIR, method, variant)
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    if not files:
        return None
    mask = A.ruler_set_mask(df, ruler_set)
    tgt, valid = A.target_values(df, target)
    sel = mask & valid
    is_place = (target == "geo")
    is_test = A.is_test_split(df, sel)[sel]
    y = tgt[sel]

    per_layer, best = {}, (None, -np.inf)
    for path in files:
        li = int(re.search(r"layer(\d+)\.npz$", path).group(1))
        X = np.load(path)["acts"][sel]
        if np.isnan(X).any():
            continue
        scores, probe, proj = probing.run_probe(X, y, is_test, is_place)
        per_layer[li] = scores
        if scores["test"]["r2"] > best[1]:
            best = (li, scores["test"]["r2"])
    if not per_layer:
        return None
    bl, br2 = best
    bsp = per_layer[bl]["test"].get(
        "spearman",
        (per_layer[bl]["test"].get("lat_spearman", float("nan"))
         + per_layer[bl]["test"].get("lon_spearman", float("nan"))) / 2)
    out = {
        "method": method, "variant": variant, "ruler_set": ruler_set,
        "target": target, "site": site,
        "n": int(sel.sum()), "n_test": int(is_test.sum()),
        "n_rulers": int(df[sel].ruler.nunique()),
        "layers": {str(k): v for k, v in sorted(per_layer.items())},
        "best_layer": bl, "best_test_r2": float(br2), "best_test_spearman": float(bsp),
    }
    pdir = os.path.join(RESULTS_DIR, "probes", method)
    os.makedirs(pdir, exist_ok=True)
    fn = f"{variant}.{ruler_set}.{target}.{site}.ridge.json"
    with open(os.path.join(pdir, fn), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{method}/{variant}/{ruler_set}/{target}/{site}] "
          f"best L{bl} test r2={br2:.3f} rho={bsp:.3f} (n={out['n']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    df = A.load_fragments()

    variants = [v for v in A.TEXT_VARIANTS
                if os.path.isdir(os.path.join(ACTS_DIR, args.method, v))]
    ok = True
    for variant in variants:
        for ruler_set in A.RULER_SETS:
            for target in A.TARGETS:
                for site in SITES:
                    if probe_one(args.method, variant, ruler_set, target,
                                 site, df, args) is None:
                        ok = False
    if args.cleanup and ok and variants:
        n = 0
        for p in glob.glob(os.path.join(ACTS_DIR, args.method, "*", "*.npz")):
            os.remove(p); n += 1
        print(f"[cleanup] removed {n} npz for {args.method}")


if __name__ == "__main__":
    main()
