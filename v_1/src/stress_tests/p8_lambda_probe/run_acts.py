"""P8 cluster runner — the lambda-dial probe on stored model activations (CPU).

SETUP   mean-pooled acts, tier0 + maximal cleanings, every stored layer;
        200 balanced draws x GroupKFold-by-ruler (standard MC).
PROBE   lambda-probe (MATH_NOTES.md): lambda grid 0..1, d=3, k=10 neighbors.
METRIC  align1 = |Spearman(leading coord, year)| held-out; pred = ridge-on-Z_d
        Spearman. Best layer surfaced twice: by align1 at lambda=1 (the
        unsupervised question) and by pred at lambda=0 (the supervised one).

Usage:  python run_acts.py --method qwen3_8b [--cleanings tier0,maximal]
                           [--k 10] [--n-jobs 16]
Writes  results/p8_lambda__{method}.json
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
from lambda_probe import mc_lambda_probe, LAMBDAS                    # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"


def _score(pl, lam, key):
    v = pl.get(f"{lam:.1f}", {}).get(f"{key}_mean", float("nan"))
    return v if v == v else -9.0


def run(args):
    df = pd.read_parquet(PARQUET)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    dm = np.load(BAL / "draws_matrix.npy")[: args.n_draws]
    draw_rows = [np.where(r)[0] for r in dm]

    out = {"method": args.method, "protocol": "p8_lambda_mc",
           "k_neighbors": args.k, "d": args.d, "lambdas": LAMBDAS,
           "cleanings": {}}
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
            r = mc_lambda_probe(load_layer(d, L), year, ruler, draw_rows,
                                k=args.k, d=args.d, l2_normalize=True)
            print(f"    L{L:02d} done ({time.time()-t0:.0f}s)", flush=True)
            return L, r

        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(L) for L in layers))
        valid = {L: r for L, r in res.items() if not r.get("skipped")}
        if not valid:
            out["cleanings"][cl] = {"skipped": True}
            continue
        b_un = max(valid, key=lambda L: _score(valid[L]["per_lambda"], 1.0, "align1"))
        b_su = max(valid, key=lambda L: _score(valid[L]["per_lambda"], 0.0, "pred"))
        blk = {"per_layer": {str(L): r for L, r in res.items()},
               "best_layer_unsup": b_un, "best_layer_sup": b_su,
               "curves": {f"L{b_un} (best @λ=1 align1)": valid[b_un]["per_lambda"],
                          f"L{b_su} (best @λ=0 pred)": valid[b_su]["per_lambda"]}}
        out["cleanings"][cl] = blk
        pu, ps = valid[b_un]["per_lambda"], valid[b_su]["per_lambda"]
        print(f"  {cl}: unsup-best L{b_un} align1@1.0="
              f"{_score(pu, 1.0, 'align1'):.3f} (pred@0.0={_score(pu, 0.0, 'pred'):.3f})"
              f" | sup-best L{b_su} pred@0.0={_score(ps, 0.0, 'pred'):.3f}"
              f" (align1@1.0={_score(ps, 1.0, 'align1'):.3f})", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p8_lambda__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--cleanings", default="tier0,maximal")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    run(p.parse_args())
