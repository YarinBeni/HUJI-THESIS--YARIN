"""J6 (MC) — P1 year-probe under the Monte-Carlo BALANCED protocol (comparable to
the thesis's maximal-balanced PLS Spearman headline).

Sites:
  mean       — <method>_{tier0,maximal}_mean    (both cleanings)
  king_last  — <method>_tier0_kinglast          (tier0 ONLY; maximal strips names)
  king_mean  — <method>_tier0_kingmean          (tier0 ONLY)

For each site x layer: 200 balanced draws (draws_matrix.npy), GroupKFold-by-ruler
PLS within each draw, best-k Spearman, averaged over draws. King sites intersect
each draw with the name-found mask. Emits results/mc/p1_year_mc__<method>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers   # noqa: E402
from mc_probe import mc_year_probe, draws_to_rows                    # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"
SUBSET = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"


def _process_site(name, acts_dir, draws, year, ruler, valid_mask, n_jobs):
    from joblib import Parallel, delayed
    layers = available_layers(acts_dir)
    draw_rows = draws_to_rows(draws, valid_mask)

    def one(L):
        X = load_layer(acts_dir, L)
        return L, mc_year_probe(X, year, ruler, draw_rows)
    res = dict(Parallel(n_jobs=n_jobs)(delayed(one)(L) for L in layers))
    per = {str(L): res[L] for L in layers}
    valid = {L: r for L, r in res.items() if not r.get("skipped")}
    if not valid:
        return {"site": name, "insufficient": True, "per_layer": per}
    bestL = max(valid, key=lambda L: valid[L]["spearman_mean"])
    return {"site": name, "best_layer": bestL,
            "best": valid[bestL], "per_layer": per}


def run(args):
    draws = np.load(args.draws)
    order = json.loads(Path(args.fragment_order).read_text())
    df = pd.read_parquet(args.corpus)
    assert len(order) == len(df), "fragment_order != corpus length"
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()

    out = {"method": args.method, "protocol": "mc_balanced", "n_draws": int(draws.shape[0]),
           "sites": {}}
    # mean (tier0 + maximal), full coverage
    for clean in ["tier0", "maximal"]:
        d = find_acts_dir(args.method, clean, "mean")
        key = f"mean_{clean}"
        out["sites"][key] = ({"missing": True} if d is None else
                             _process_site(key, d, draws, year, ruler, None, args.n_jobs))
    # king sites (tier0 only), partial coverage
    for suffix, key in [("kinglast", "king_last"), ("kingmean", "king_mean")]:
        d = ACTS / f"{args.method}_tier0_{suffix}"
        cov = d / "king_coverage.json"
        if not cov.exists() or not any(d.glob("layer_*.npz")):
            out["sites"][key] = {"missing": True}; continue
        found = np.array(json.loads(cov.read_text())["found"], dtype=bool)
        out["sites"][key] = _process_site(key, d, draws, year, ruler, found, args.n_jobs)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p1_year_mc__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, s in out["sites"].items():
        if s.get("missing") or s.get("insufficient"):
            print(f"  {k:13s}: {'missing' if s.get('missing') else 'insufficient'}")
        else:
            b = s["best"]
            rg = b.get("ridge", {}).get("spearman_mean", float("nan"))
            print(f"  {k:13s}: L{s['best_layer']} PLS(k={b.get('best_k')})={b['spearman_mean']:.3f}±{b['spearman_std']:.2f} "
                  f"ridge={rg:.3f} null={b['shuffled_spearman_mean']:.3f} mae={b['mae_mean']:.0f} "
                  f"(draws={b['n_draws_used']})")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--draws", default=str(SUBSET / "draws_matrix.npy"))
    p.add_argument("--fragment-order", default=str(SUBSET / "corpus_fragment_order.json"))
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "mc"))
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
