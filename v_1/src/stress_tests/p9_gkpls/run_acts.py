"""P9 cluster runner — G-KPLS (+ RBF-KPLS, KRR-on-K_G baselines) on stored
mean activations, all four cleanings.

SETUP   mean-pooled acts, {tier0, maximal, maxking, engtier0}, every layer;
        200 balanced draws x GroupKFold-by-ruler.
PROBE   G-KPLS a in {1,2,3,5} best-a; RBF-KPLS (isolates geodesic vs kernel);
        KRR on K_G, lam in {1e-3,1e-2,1e-1} (isolates PLS vs kernel); k=10 graph.
METRIC  Spearman(predicted year, true year), mean +- std over draws.

Usage:  python run_acts.py --method qwen3_8b [--n-jobs 16]
Writes  results/p9_gkpls__{method}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from gkpls import mc_gkpls_probe                                     # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]
ARMS = ("gkpls", "rbfkpls", "krr_geo")


def run(args):
    df = pd.read_parquet(PARQUET)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    dm = np.load(BAL / "draws_matrix.npy")[: args.n_draws]
    draw_rows = [np.where(r)[0] for r in dm]

    out = {"method": args.method, "protocol": "p9_gkpls_mc",
           "k_neighbors": args.k, "cleanings": {}}
    for cl in args.cleanings.split(","):
        d = find_acts_dir(args.method, cl, "mean")
        if d is None:
            print(f"[{args.method} x {cl}] acts missing — skipped", flush=True)
            out["cleanings"][cl] = {"missing": True}
            continue
        layers = available_layers(d)
        print(f"[{args.method} x {cl}] {len(layers)} layers from {d}", flush=True)

        from joblib import Parallel, delayed

        def one(L):
            t0 = time.time()
            r = mc_gkpls_probe(load_layer(d, L), year, ruler, draw_rows,
                               k=args.k)
            print(f"    L{L:02d} done ({time.time()-t0:.0f}s)", flush=True)
            return L, r

        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(L) for L in layers))
        valid = {L: r for L, r in res.items() if not r.get("skipped")}
        if not valid:
            out["cleanings"][cl] = {"skipped": True}
            continue
        bL = max(valid, key=lambda L: valid[L]["gkpls"]["spearman_mean"]
                 if valid[L]["gkpls"]["spearman_mean"] == valid[L]["gkpls"]["spearman_mean"] else -9)
        blk = {"per_layer": {str(L): r for L, r in res.items()},
               "best_layer": bL,
               "best": {arm: valid[bL][arm] for arm in ARMS}}
        out["cleanings"][cl] = blk
        b = blk["best"]
        print(f"  {cl}: L{bL}  gkpls={b['gkpls']['spearman_mean']:.3f}"
              f"(a={b['gkpls']['best_hyper']})"
              f"  rbfkpls={b['rbfkpls']['spearman_mean']:.3f}"
              f"  krr_geo={b['krr_geo']['spearman_mean']:.3f}", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p9_gkpls__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--cleanings", default=",".join(CLEANINGS))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    run(p.parse_args())
