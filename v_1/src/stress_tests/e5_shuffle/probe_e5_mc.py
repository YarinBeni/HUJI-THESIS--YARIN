"""E5 probe (CPU) — balanced-MC year Spearman on shuffled vs unshuffled
embeddings, extracted by extract_shuffled_acts.py with identical settings.

SETUP   4 cleanings (tier0/maximal/maxking/engtier0) x 2 variants
        (unshuf = word-capped original order, shuf = same words permuted),
        mean pool, 8 rulers x 21 x 200 balanced draws, GroupKFold-by-ruler.
PROBE   PLS (k in {1,2,3,5}, best-k) + Ridge.
METRIC  Spearman(predicted year, true year). The number that matters is the
        DELTA unshuf - shuf at each variant's own best layer:
          delta ~ 0  -> word order contributes nothing; the probe signal is
                        bag-of-tokens (TF-IDF-like), not composition.
          delta >> 0 -> order does matter; the embedding encodes more than
                        lexical identity.

Usage:  python probe_e5_mc.py --method qwen3_8b [--n-jobs 8]
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

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from mc_probe import mc_year_probe, draws_to_rows                    # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DRAWS = (_REPO / "v_1/src/linear_probing/results/orcc_round2_phase0"
         / "balanced_subset/draws_matrix.npy")
CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]


def probe_dir(d, year, ruler, draw_rows, n_jobs):
    from joblib import Parallel, delayed
    layers = available_layers(d)

    def one(L):
        return L, mc_year_probe(load_layer(d, L), year, ruler, draw_rows)
    res = dict(Parallel(n_jobs=n_jobs)(delayed(one)(L) for L in layers))
    valid = {L: r for L, r in res.items() if not r.get("skipped")}
    if not valid:
        return {"skipped": True}
    bL = max(valid, key=lambda L: valid[L]["spearman_mean"])
    return {"per_layer": {str(L): r for L, r in res.items()},
            "best_layer": bL,
            "best": {k: valid[bL][k] for k in
                     ("best_k", "spearman_mean", "spearman_std", "ridge")}}


def run(args):
    df = pd.read_parquet(CORPUS)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    draw_rows = draws_to_rows(np.load(DRAWS))

    out = {"method": args.method, "protocol": "e5_word_shuffle_mc",
           "metric": "spearman", "cleanings": {}}
    for cl in CLEANINGS:
        blk = {}
        for var in ("unshuf", "shuf"):
            d = find_acts_dir(args.method, f"{var}{cl}", "mean")
            blk[var] = ({"missing": True} if d is None else
                        probe_dir(d, year, ruler, draw_rows, args.n_jobs))
        u, s = blk["unshuf"].get("best"), blk["shuf"].get("best")
        if u and s:
            blk["delta_spearman"] = u["spearman_mean"] - s["spearman_mean"]
            blk["delta_ridge"] = (u["ridge"]["spearman_mean"]
                                  - s["ridge"]["spearman_mean"])
        out["cleanings"][cl] = blk

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"e5_mc__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        u, s = blk["unshuf"].get("best"), blk["shuf"].get("best")
        if not (u and s):
            print(f"  {cl:9s}: missing ({'unshuf' if not u else 'shuf'})"); continue
        print(f"  {cl:9s}: unshuf {u['spearman_mean']:.3f}±{u['spearman_std']:.2f}"
              f" (R {u['ridge']['spearman_mean']:.3f}, L{blk['unshuf']['best_layer']})"
              f"  |  shuf {s['spearman_mean']:.3f}±{s['spearman_std']:.2f}"
              f" (R {s['ridge']['spearman_mean']:.3f}, L{blk['shuf']['best_layer']})"
              f"  |  delta {blk['delta_spearman']:+.3f}"
              f" (R {blk['delta_ridge']:+.3f})")
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    p.add_argument("--n-jobs", type=int, default=8)
    run(p.parse_args())
