"""WB-PLS — best-layer PLS-k sweep for the CELL-B entity surfaces.

`probe_entity.py` reports PLS at a single k=5. The fragment-level and cell-A entity
plots sweep k = 1..64, so this adds the matching sweep for `assyrian_ruler` and
`mesopotamian_place`, making the PLS figure comparable across all four surfaces:

    entity level    cell A (paper, salient)  -> results/eng_pls/
    entity level    cell B (ours,  obscure)  -> probes_entity_pls/     [this file]
    fragment level  cell B (eng_tier0)       -> akkadian/results/layers_pls/
    fragment level  cell C (akk_maximal)     -> akkadian/results/layers_pls/

Best layer is reused from the committed probe_entity JSON (no re-sweep), so this is
seconds per arm. Scoring is the same entity-level MC (200 draws, 20% of ENTITIES held
out) so a ruler never straddles train/test.

    python probe_entity_pls.py --method qwen3_8b
"""
from __future__ import annotations

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
import probe_entity as PE                              # noqa: E402
from wm_lib import probing                             # noqa: E402

KS = [1, 2, 3, 5, 8, 16, 32, 64]
OUT = os.path.join(_HERE, "results", "probes_entity_pls")


def _best_layer(method, et, site):
    fp = os.path.join(PE.RESULTS_DIR, "probes_entity", method, f"{et}.{site}.json")
    if not os.path.exists(fp):
        return None
    return json.load(open(fp)).get("best_layer")


def probe_one(method, et, site, args):
    act_dir = os.path.join(PE.ACTS_DIR, method, et)
    files = {int(re.search(r"layer(\d+)\.npz$", p).group(1)): p
             for p in glob.glob(os.path.join(act_dir, f"{site}.layer*.npz"))}
    if not files:
        return None
    bl = _best_layer(method, et, site)
    if bl is None or bl not in files:
        bl = sorted(files)[len(files) // 2]

    df = PE.load_df(et)
    y, is_place = PE.targets(et, df)
    ent_ix = df["entity_ix"].values
    X = np.load(files[bl])["acts"].astype(np.float32)
    X, bad = probing.sanitize(X)
    if bad > 0.01:
        print(f"[warn] {method}/{et}/{site} L{bl}: {bad:.1%} non-finite", flush=True)

    out = {"method": method, "entity_type": et, "site": site, "best_layer": int(bl),
           "n_entities": int(len(np.unique(ent_ix))), "ks": KS, "rows": {}}
    for tag in ("bare", "all"):
        m = (df["template"].values == "bare") if tag == "bare" else np.ones(len(df), bool)
        if m.sum() < 10:
            continue
        res = {}
        for k in KS:
            if k >= min(m.sum(), X.shape[1]):
                continue
            try:
                res[str(k)] = PE.mc_entity_scores(X[m], y[m], ent_ix[m], is_place,
                                                  n_draws=args.n_draws, k=k)
            except Exception as e:                              # noqa: BLE001
                res[str(k)] = {"error": f"{type(e).__name__}: {e}"[:100]}
        ok = {k: v for k, v in res.items() if "mc_rho" in v}
        bk = max(ok, key=lambda k: ok[k]["mc_rho"], default=None)
        out["rows"][tag] = {"pls_by_k": res, "best_k": int(bk) if bk else None,
                            "n": int(m.sum())}
    d = os.path.join(OUT, method)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{et}.{site}.json"), "w") as f:
        json.dump(out, f, indent=2)
    b = out["rows"].get("bare", {})
    print(f"[{method}/{et}/{site}] L{bl} best_k(bare)={b.get('best_k')}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n-draws", type=int, default=100)
    args = ap.parse_args()
    for et in PE.ENTITY_TYPES:
        for site in PE.SITES:
            try:
                probe_one(args.method, et, site, args)
            except Exception as e:                              # noqa: BLE001
                print(f"[error] {args.method}/{et}/{site}: {type(e).__name__}: {e}",
                      flush=True)


if __name__ == "__main__":
    main()
