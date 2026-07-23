"""WA probing: three modes over the Akkadian activations for one method.

For each text variant on disk × ruler set × target × pooling site, we report:
  * holdout — within-ruler 80/20 split, per layer, best layer (the original, biased
              by ruler identity; kept for continuity / G&T-comparability).
  * mc      — balanced Monte-Carlo (cap per ruler, 200 draws, StratifiedKFold-by-ruler)
              at the holdout-best layer: in-distribution, imbalance removed.
  * loro    — leave-one-ruler-out (pooled OOF) swept over layers, best by Spearman:
              the real "place an unseen ruler" generalization test.

Writes results/probes/{method}/{variant}.{ruler_set}.{target}.{site}.json (committed).

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
import akk_modes as M                         # noqa: E402
from wm_lib import probing                    # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
SITES = ["last", "mean"]
MC_DRAWS = 200


def _load_layers(act_dir, site, sel):
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    layers = {}
    for path in files:
        li = int(re.search(r"layer(\d+)\.npz$", path).group(1))
        X = np.load(path)["acts"][sel].astype(np.float32)
        if not np.isnan(X).any():
            layers[li] = X
    return layers


def probe_one(method, variant, ruler_set, target, site, df, args):
    act_dir = os.path.join(ACTS_DIR, method, variant)
    mask = A.ruler_set_mask(df, ruler_set)
    tgt, valid = A.target_values(df, target)
    sel = mask & valid
    is_place = (target == "geo")
    y = tgt[sel]
    ruler = df.ruler.values[sel]
    layers = _load_layers(act_dir, site, sel)
    if not layers:
        return None

    # --- holdout (per layer, within-ruler 80/20) ---
    is_test = A.is_test_split(df, sel)[sel]
    hold, best = {}, (None, -np.inf)
    for li, X in layers.items():
        scores, _, _ = probing.run_probe(X, y, is_test, is_place)
        hold[li] = scores
        if scores["test"]["r2"] > best[1]:
            best = (li, scores["test"]["r2"])
    bl = best[0]
    bsp = hold[bl]["test"].get(
        "spearman", (hold[bl]["test"].get("lat_spearman", np.nan)
                     + hold[bl]["test"].get("lon_spearman", np.nan)) / 2)

    # --- balanced-MC at the holdout-best layer ---
    mc = M.mc_balanced(layers[bl], y, ruler, is_place, n_draws=MC_DRAWS)
    mc["layer"] = bl

    # --- leave-one-ruler-out, swept over layers, best by Spearman ---
    loro_best = {"spearman": -np.inf}
    for li, X in layers.items():
        r = M.loro(X, y, ruler, is_place)
        if r.get("skipped"):
            continue
        if r["spearman"] > loro_best["spearman"]:
            loro_best = {**r, "layer": li}

    out = {
        "method": method, "variant": variant, "ruler_set": ruler_set,
        "target": target, "site": site,
        "n": int(sel.sum()), "n_rulers": int(len(np.unique(ruler))),
        "holdout": {"best_layer": bl, "best_test_r2": float(best[1]),
                    "best_test_spearman": float(bsp), "n_test": int(is_test.sum())},
        "mc": mc,
        "loro": loro_best if loro_best["spearman"] > -np.inf else {"skipped": True},
        # back-compat with the first aggregator
        "best_layer": bl, "best_test_r2": float(best[1]),
        "best_test_spearman": float(bsp),
    }
    pdir = os.path.join(RESULTS_DIR, "probes", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{variant}.{ruler_set}.{target}.{site}.ridge.json"), "w") as f:
        json.dump(out, f, indent=2)
    lo = out["loro"].get("spearman", float("nan"))
    print(f"[{method}/{variant}/{ruler_set}/{target}/{site}] "
          f"holdout r2={best[1]:.3f} | mc rho={mc['spearman_mean']:.3f}"
          f"±{mc['spearman_std']:.3f} | loro rho={lo:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default=None,
                    help="probe only this text variant (default: all on disk)")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    df = A.load_fragments()
    variants = [v for v in A.TEXT_VARIANTS
                if os.path.isdir(os.path.join(ACTS_DIR, args.method, v))
                and (args.variant is None or v == args.variant)]
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
