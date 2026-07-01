"""J13 — probe the "maximal-with-kings" activations.

Three pooling sites, all on the maximal_keepking cleaning (apples-to-apples):
    mean       -> {method}_maxking_mean
    king_last  -> {method}_maxking_kinglast
    king_mean  -> {method}_maxking_kingmean

For each site x layer, the balanced-MC engine (mc_maxking) runs year_group (legacy
GroupKFold Spearman), year_strat (StratifiedKFold Spearman/MAE/+-10yr accuracy) and
the ruler_clf control (StratifiedKFold macro-F1 vs chance/shuffle). Best layer is
chosen by ruler_clf macro-F1. Draws come from the 5-ruler / k=9 king-found subset.

Emits results/maxking/p1_maxking__{method}.json.
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
from geo_loader import load_layer, available_layers   # noqa: E402
from mc_maxking import mc_maxking_probe                # noqa: E402
from mc_probe import draws_to_rows                     # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"
SUBSET = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset_maxking"

SITES = [("mean", "mean"), ("king_last", "kinglast"), ("king_mean", "kingmean")]


def _process_site(acts_dir, draws, year, ruler, found, n_jobs):
    from joblib import Parallel, delayed
    layers = available_layers(acts_dir)
    draw_rows = draws_to_rows(draws, found)

    def one(L):
        return L, mc_maxking_probe(load_layer(acts_dir, L), year, ruler, draw_rows)
    res = dict(Parallel(n_jobs=n_jobs)(delayed(one)(L) for L in layers))
    per = {str(L): res[L] for L in layers}
    valid = {L: r for L, r in res.items() if not r.get("skipped")}
    if not valid:
        return {"insufficient": True, "per_layer": per}
    bestL = max(valid, key=lambda L: valid[L]["ruler_clf"]["macro_f1_mean"])
    return {"best_layer": bestL, "best": valid[bestL], "per_layer": per}


def run(args):
    draws = np.load(args.draws)
    order = json.loads(Path(args.fragment_order).read_text())
    df = pd.read_parquet(args.corpus)
    assert len(order) == len(df), "fragment_order != corpus length"
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()

    out = {"method": args.method, "protocol": "mc_balanced_maxking",
           "config": "maximal-with-kings", "n_draws": int(draws.shape[0]), "sites": {}}
    for site, suffix in SITES:
        d = ACTS / f"{args.method}_maxking_{suffix}"
        cov = d / "king_coverage.json"
        if not cov.exists() or not any(d.glob("layer_*.npz")):
            out["sites"][site] = {"missing": True}
            continue
        found = np.array(json.loads(cov.read_text())["found"], dtype=bool)
        out["sites"][site] = _process_site(d, draws, year, ruler, found, args.n_jobs)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p1_maxking__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for site, s in out["sites"].items():
        if s.get("missing") or s.get("insufficient"):
            print(f"  {site:10s}: {'missing' if s.get('missing') else 'insufficient'}")
            continue
        b = s["best"]; rc = b["ruler_clf"]; ys = b["year_strat"]
        acc10 = ys["per_k"][str(ys["best_k"])]["acc10_mean"]
        print(f"  {site:10s}: L{s['best_layer']} rulerF1={rc['macro_f1_mean']:.3f}"
              f"(chance={rc['chance_macro_f1']:.2f},shuf={rc['shuffled_macro_f1']:.2f}) "
              f"| year_strat sp={ys['spearman_mean']:.3f} acc±10={acc10:.2f} "
              f"| year_group sp={b['year_group']['spearman_mean']:.3f} (draws={b['n_draws_used']})")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--draws", default=str(SUBSET / "draws_matrix.npy"))
    p.add_argument("--fragment-order", default=str(SUBSET / "corpus_fragment_order.json"))
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "maxking"))
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
