"""E-prop — the experiment F11 opened: if the year features fire MID-TEXT on
English documents but the signal never reaches the last token, can we recover
document-level time by pooling the features over tokens ourselves?

DESIGN. Stage 1 (GPU): re-run the fragments through Qwen3-8B, encode every
token through the Scope SAE at the F8 layer, and MAX-POOL each candidate year
feature over the fragment's tokens — a (n_fragments x n_features) matrix that
represents "how strongly did this feature fire anywhere in this text". Stage 2
(CPU): run the exact E1 pairwise protocol on that matrix.

WHAT EACH OUTCOME MEANS. If max-pooled SAE features beat the .586 floor on the
English gloss, the mid-text signal is REAL usable chronology and the collapse
is specifically a propagation failure — which is both the mechanism and a
practical partial fix ("read the features yourself instead of trusting the last
token"). If they do not beat the floor, the mid-text firings are temporal noise
(fires on date-like tokens without carrying the document's date).

Candidates = features firing on >=2% of cell-A entities (the F8 candidate rule,
recomputed here since F8 only persisted its top-50).

    python propagation_probe.py                # stage 1 then stage 2
    python propagation_probe.py --stage 2      # reuse the saved matrix

Writes results/propagation.{variant}.json + the feature matrix npz. GPU (F18).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fvu_gate import ENT_ACTS, ENTITY, METHOD, RESULTS, load_layer_acts, load_sae  # noqa: E402
from token_firing import token_features                     # noqa: E402

_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                      # noqa: E402
import probe_pairs as PP                                    # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", f"{ENTITY}.csv")

VARIANTS = {"eng_tier0": "text_eng_tier0", "akk_maximal": "text_akk"}


def candidates(gate):
    """F8's candidate rule, recomputed: features firing >=2% on cell-A last."""
    import torch
    L = min(gate["passing_layers"],
            key=lambda l: gate["fvu"][str(l)]["cellA_entities"])
    off = gate["offset"]
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, ENTITY), L + off)
    W_enc, b_enc, _, _ = load_sae(L)
    pre = torch.from_numpy(Xa).float() @ W_enc.T + b_enc
    val, idx = torch.topk(pre, 100, dim=-1)
    z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val)).numpy()
    fire = (z > 0).mean(0)
    return L, sorted(np.where(fire >= 0.02)[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                    help="0 = both")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    gate = json.load(open(os.path.join(RESULTS, "fvu_gate.json")))
    L, feats = candidates(gate)
    print(f"[cand] SAE block {L}: {len(feats)} candidate features", flush=True)
    df = P.load_eligible()
    mat_p = os.path.join(RESULTS, f"prop_features.layer{L}.npz")

    if args.stage in (0, 1):
        mats = {}
        for var, col in VARIANTS.items():
            texts = df[col].fillna("").astype(str).tolist()
            print(f"[stage1] {var}: {len(texts)} fragments", flush=True)
            fired, maxact, _ = token_features(texts, L, feats,
                                              args.batch, 512)
            mats[var] = maxact.astype(np.float32)     # (n, n_feats) max-pooled
        np.savez_compressed(mat_p, feats=np.array(feats),
                            **{v: m for v, m in mats.items()})
        print(f"[stage1] -> {mat_p}", flush=True)

    if args.stage in (0, 2):
        z = np.load(mat_p)
        for var in VARIANTS:
            X = z[var]

            def get_feats(tr_pos, X=X):
                from sklearn.preprocessing import StandardScaler
                pos2row = {p: i for i, p in enumerate(df.pos.values)}
                rows = np.array([pos2row[p] for p in tr_pos])
                sc = StandardScaler().fit(X[rows])
                return lambda pos: sc.transform(
                    X[np.array([pos2row[p] for p in pos])])
            per = PP.run_mc(df, get_feats, args.m, args.draws, args.seed,
                            P.eligible_ruler_pairs(df))
            s = PP.summarize(per)
            out = {"variant": var, "sae_block": L, "n_features": len(feats),
                   "pooling": "max-over-tokens", "m": args.m,
                   "draws": args.draws, "full": s}
            pth = os.path.join(RESULTS, f"propagation.{var}.json")
            with open(pth, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[stage2] {var}: macro={s.get('macro_acc_mean', float('nan')):.3f}"
                  f"±{s.get('macro_acc_std', float('nan')):.3f} -> {pth}",
                  flush=True)


if __name__ == "__main__":
    main()
