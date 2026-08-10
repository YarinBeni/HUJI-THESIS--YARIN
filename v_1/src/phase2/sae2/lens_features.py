"""SAE2 step 3' (job F24) — self-made labels via decoder-row logit lens.

Neuronpedia turned out to host ONLY layer 18 of the Karvonen release
(confirmed by API: the layer-9 source 404s where layer-18 answers), and
layer 18 fails our FVU gate — so autointerp labels are unavailable for the
usable instrument. This is the fallback the plan itself named: lens each top
feature's DECODER ROW through W_U (the E4.4 machinery) and read which tokens
it points at, plus a keyword taxonomy pass over those tokens. Not autointerp,
but an intrinsic, replicable read of what each feature writes to the vocab.

    python lens_features.py            # top-10 hunt features
    python lens_features.py --features 22835 44713

Writes results/feature_lens.layer{L}.json. CPU (~16GB for the unembedding).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import karvonen as K                                     # noqa: E402
from fetch_labels import classify_label                  # noqa: E402
_TRACES = os.path.abspath(os.path.join(_HERE, "..", "traces"))
sys.path.insert(0, _TRACES)
from logit_lens import load_unembed, top_tokens          # noqa: E402
_SAE1 = os.path.abspath(os.path.join(_HERE, "..", "sae"))
sys.path.insert(0, _SAE1)
from fvu_gate import METHOD                              # noqa: E402

RESULTS = os.path.join(_HERE, "results")


def classify_tokens(entries):
    """Taxonomy vote over a token list: which categories the lens end hits."""
    hits = {}
    for e in entries:
        cat = classify_label(e["token"])
        if cat != "other":
            hits.setdefault(cat, []).append(e["token"])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, nargs="*", default=None)
    ap.add_argument("--top-feats", type=int, default=10)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    pipe = json.load(open(os.path.join(RESULTS, "pipeline.json")))
    repo, L = pipe["step0"]["repo"], pipe["step0"]["layer_used"]
    sae = K.load(repo, pipe["step0"]["file_used"])
    W_dec = sae["W_dec"].numpy()
    tab = pd.read_csv(sorted(glob.glob(os.path.join(
        RESULTS, "feature_hunt2.layer*.csv")))[-1])
    feats = (args.features if args.features
             else tab.feature.astype(int).head(args.top_feats).tolist())

    tok, W_U, norm = load_unembed(METHOD)
    out = {"layer": L, "d_sae": sae["d_sae"], "features": {}}
    for f in feats:
        ends = top_tokens(W_dec[int(f)], W_U, norm, tok, args.topk)
        row = tab[tab.feature == f]
        rec = {"rho_year": (float(row.rho_year.iloc[0]) if len(row) else None),
               "positive_end": ends["positive_end"],
               "negative_end": ends["negative_end"],
               "taxonomy_pos": classify_tokens(ends["positive_end"]),
               "taxonomy_neg": classify_tokens(ends["negative_end"])}
        out["features"][int(f)] = rec
        toks = [e["token"] for e in ends["positive_end"][:8]]
        print(f"[{f}] rho={rec['rho_year']} pos8={toks} "
              f"tax+={list(rec['taxonomy_pos'])}", flush=True)

    path = os.path.join(RESULTS, f"feature_lens.layer{L}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[done] -> {path}", flush=True)


if __name__ == "__main__":
    main()
