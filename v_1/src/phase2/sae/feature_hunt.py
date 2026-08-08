"""E5.2 — feature hunt: which SAE features carry the year, and do they fire on
Akkadian at all?

Runs only on layers that passed the FVU gate (fvu_gate.json). Three questions:

  1. Which features correlate with the entity's year on cell A?  Ranked by
     |Spearman(feature activation, death year)| over held-out entities, among
     features that fire on >= --min-fire of entities.
  2. Which features ALIGN with the ridge year direction?  cos(W_dec row, coef).
     Correlation and alignment are different claims; both columns are kept.
  3. Do the top year-features fire on fragments at all?  Their firing rate on
     the English gloss and on raw Akkadian, next to their cell-A rate. Silence
     on Akkadian = the collapse reaches the feature basis (a phase-2 result in
     itself); firing without year signal = present but unused.

    python feature_hunt.py                 # best gated layer, top 50
    python feature_hunt.py --layer 20

Writes results/feature_hunt.layer{L}.json (+ .csv table for eyeballing).
Feature descriptions: look ids up in the QwenScope Space / the Chongrong-Nathan
metadata — ids here are Scope feature indices at that layer.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fvu_gate import (AKK_ACTS, ENT_ACTS, ENTITY, K, METHOD,   # noqa: E402
                      RESULTS, load_layer_acts, load_sae)

_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
DIRS_A = os.path.join(_WM, "results", "directions")
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", f"{ENTITY}.csv")


def encode(X, sae, k=K, batch=2048):
    """TopK feature activations, dense (n, d_sae) float32 (zeros elsewhere)."""
    import torch
    W_enc, b_enc, W_dec, b_dec = sae
    outs = []
    Xt = torch.from_numpy(X).float()
    for i in range(0, len(Xt), batch):
        pre = Xt[i:i + batch] @ W_enc.T + b_enc
        val, idx = torch.topk(pre, k, dim=-1)
        z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
        outs.append(z)
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="SAE block index; default = gated layer with best cell-A fvu")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-fire", type=float, default=0.02)
    args = ap.parse_args()

    gate_p = os.path.join(RESULTS, "fvu_gate.json")
    if not os.path.exists(gate_p):
        sys.exit("run fvu_gate.py first — the layer offset and the gate verdict "
                 "come from its output.")
    gate = json.load(open(gate_p))
    offset = gate["offset"]
    passing = gate["passing_layers"]
    if not passing:
        sys.exit(f"no layer passed the FVU gate ({gate['fvu']}) — the Scope SAE "
                 "cannot represent these activations; do not interpret features.")
    L = args.layer if args.layer is not None else min(
        passing, key=lambda l: gate["fvu"][str(l)]["cellA_entities"])
    if L not in passing:
        print(f"[warn] layer {L} did not pass the gate "
              f"(fvu {gate['fvu'].get(str(L))}) — results are exploratory",
              flush=True)

    ent = pd.read_csv(ENT_CSV)
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, ENTITY), L + offset)
    if Xa is None or len(Xa) != len(ent):
        sys.exit(f"entity acts missing/mismatched at our layer {L + offset}")
    year = ent["death_year"].values.astype(float)
    test = ent["is_test"].astype(bool).values
    ok = np.isfinite(year) & test
    sae = load_sae(L)
    Z = encode(Xa, sae)
    print(f"[encode] cell A: {Z.shape}, held-out n={ok.sum()}", flush=True)

    fire = (Z[ok] > 0).mean(0)
    cand = np.where(fire >= args.min_fire)[0]
    rho = np.zeros(len(cand))
    for i, f in enumerate(cand):
        rho[i] = stats.spearmanr(Z[ok, f], year[ok]).correlation
    coef = None
    g = sorted(glob.glob(os.path.join(DIRS_A, METHOD, f"{ENTITY}.*.layer*.npz")))
    if g:
        coef = np.load(g[0])["coef"].astype(np.float32).ravel()
        W_dec = sae[2].numpy()
        cosd = (W_dec[cand] @ coef) / (
            np.linalg.norm(W_dec[cand], axis=1) * np.linalg.norm(coef) + 1e-8)
    else:
        cosd = np.full(len(cand), np.nan)

    tab = pd.DataFrame({"feature": cand, "fire_cellA": fire[cand],
                        "rho_year": rho, "cos_ridge": cosd})
    tab["abs_rho"] = tab.rho_year.abs()
    tab = tab.sort_values("abs_rho", ascending=False).head(args.top)

    # do the year-features fire on fragments?
    for name, d in (("eng_tier0", os.path.join(AKK_ACTS, METHOD, "eng_tier0")),
                    ("akk_maximal", os.path.join(AKK_ACTS, METHOD, "akk_maximal"))):
        Xf = load_layer_acts(d, L + offset)
        if Xf is None:
            tab[f"fire_{name}"] = np.nan
            continue
        Zf = encode(Xf, sae)
        tab[f"fire_{name}"] = (Zf[:, tab.feature.values] > 0).mean(0)

    os.makedirs(RESULTS, exist_ok=True)
    csv = os.path.join(RESULTS, f"feature_hunt.layer{L}.csv")
    tab.drop(columns="abs_rho").to_csv(csv, index=False)
    summary = {
        "layer": int(L), "offset": offset, "n_candidates": int(len(cand)),
        "min_fire": args.min_fire,
        "top_abs_rho": float(tab.abs_rho.iloc[0]),
        "median_fire_ratio_akk_vs_cellA": float(
            (tab.get("fire_akk_maximal", pd.Series(dtype=float))
             / tab.fire_cellA).median()) if "fire_akk_maximal" in tab else None,
        "table": csv,
    }
    with open(os.path.join(RESULTS, f"feature_hunt.layer{L}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(tab.drop(columns="abs_rho").head(15).to_string(index=False), flush=True)
    print(f"[done] -> {csv}", flush=True)


if __name__ == "__main__":
    main()
