"""P10 cluster runner — reduce-then-{gkpls/rbfkpls/krr/dial} on stored mean acts.

For a method, at the P9-best layer of a cleaning, sweep {raw,pca,pls,umap} × {none,
zscore,l2} under the balanced-MC protocol (200 draws × GroupKFold-by-ruler). `raw`
reproduces P9/P8 (the anchor); the rest test the advisor's "reduce first" idea.

Usage:  python run_acts.py --method qwen3_8b [--cleanings maximal,engtier0]
                           [--dims 3] [--n-draws 100] [--n-jobs 12]
Writes  results/p10__{method}.json
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
import reduce_kernels as RK                                          # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
P9RES = _THIS.parents[1] / "p9_gkpls" / "results"
CLEANINGS = ["maximal", "engtier0"]


def best_layer_from_p9(method, cleaning, layers):
    """Reuse P9's best layer if available, else the middle layer."""
    fp = P9RES / f"p9_gkpls__{method}.json"
    if fp.exists():
        blk = json.loads(fp.read_text()).get("cleanings", {}).get(cleaning, {})
        if isinstance(blk.get("best_layer"), int) and blk["best_layer"] in layers:
            return blk["best_layer"]
    return layers[len(layers) // 2]


def run(args):
    df = pd.read_parquet(PARQUET)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    draw_rows = [np.where(r)[0] for r in
                 np.load(BAL / "draws_matrix.npy")[: args.n_draws]]

    configs = [(red, norm) for red in RK.REDUCERS for norm in RK.NORMS]
    out = {"method": args.method, "protocol": "p10_reduce_kernels",
           "dims": args.dims, "n_draws": args.n_draws, "cleanings": {}}

    for cl in args.cleanings.split(","):
        d = find_acts_dir(args.method, cl, "mean")
        if d is None:
            print(f"[{args.method} x {cl}] acts missing — skipped", flush=True)
            out["cleanings"][cl] = {"missing": True}
            continue
        layers = available_layers(d)
        L = best_layer_from_p9(args.method, cl, layers)
        X = load_layer(d, L)
        print(f"[{args.method} x {cl}] layer {L} (of {len(layers)}), X={X.shape}",
              flush=True)

        from joblib import Parallel, delayed

        def one(cfg):
            red, norm = cfg
            t0 = time.time()
            r = RK.mc_reduced(X, year, ruler, draw_rows, reducer=red, norm=norm,
                              dims=args.dims, do_gkpls=True, do_dial=True)
            print(f"    {red}/{norm}: gkpls="
                  f"{r.get('gkpls', {}).get('spearman_mean', float('nan')):.3f}"
                  f" ({time.time()-t0:.0f}s)", flush=True)
            return f"{red}/{norm}", r

        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(c) for c in configs))
        out["cleanings"][cl] = {"layer": L, "configs": res}

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p10__{args.method}.json"
    if fp.exists():
        prev = json.loads(fp.read_text()).get("cleanings", {})
        out["cleanings"] = {**prev, **out["cleanings"]}
    fp.write_text(json.dumps(out, indent=2))
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--cleanings", default=",".join(CLEANINGS))
    p.add_argument("--dims", type=int, default=3)
    p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    run(p.parse_args())
