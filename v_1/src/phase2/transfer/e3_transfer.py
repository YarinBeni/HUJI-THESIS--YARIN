"""E3 — frozen name-direction transfer + the LEACE mediation test.

THE QUESTION. Cell A proved a year direction exists for ENTITY NAMES (ridge on
name activations, rho ~.88). Is the document-side time axis the SAME axis, just
weaker — or a different axis? And when the frozen direction does order fragments,
does it do so THROUGH ruler identity, or independently of it?

ZERO document-side fitting. The cell-A ridge coefficient vector (saved by
probe_wm.py at its best layer, raw-activation coordinates) is applied as-is to
fragment activations: s = coef . x. Nothing is trained on fragments, so there is
nothing to leak — every fragment is evaluation data. Read-outs:

  * Spearman(s, year) across fragments;
  * the E1 pairwise evaluation with s as the scorer (macro accuracy over
    ruler-pairs, same draws protocol) — directly comparable to the E1 table,
    where every trained probe HAD to be fitted on fragments;
  * both again after LEACE erasure of one-hot ruler identity from the fragment
    activations (concept-erasure, Belrose et al. 2023). Collapse under erasure
    = the transfer was an identity lookup; survival = a ruler-independent time
    component. (By the ICC=1 degeneracy this is a MEDIATION test, not a "does
    year survive" test — see DECIDED_EXPERIMENTS.md F2.)

Layer handling: the frozen direction lives at the cell-A best layer L*. Primary
read-out applies it at the SAME residual-stream depth L* of the fragment run;
a full sweep over fragment layers is reported as exploratory.

Also computed when E1 directions exist: cosine between this frozen cell-A
direction and E1's pairwise direction (trained on relative order only, no
absolute years). Both are moved into the same standardized coordinates
(w_scaled = coef * sd_features over fragments) before the cosine; cross-layer
cosines are labeled as such.

    python e3_transfer.py --method olmo2_7b --variant akk_maximal
    python e3_transfer.py --method olmo2_7b --variant akk_maximal --skip-leace

Writes results/{method}.{variant}.{site}.json. Needs the akkadian npz store and
world_models/results/directions/{method}/ (both cluster-local).
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
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
DIRS_ROOT = os.path.join(_WM, "results", "directions")
RESULTS = os.path.join(_HERE, "results")
ENTITY = "historical_figure"      # the cell-A set whose direction transfers


def find_cellA_direction(method, dirs_root):
    """The saved best-layer ridge direction for the entity set. probe_wm saves
    one file per entity_type x site: {entity}.{site}.layer{L}.npz."""
    g = sorted(glob.glob(os.path.join(dirs_root, method,
                                      f"{ENTITY}.*.layer*.npz")))
    if not g:
        sys.exit(f"no cell-A direction for {method} under {dirs_root}/{method} "
                 f"(pattern {ENTITY}.*.layer*.npz).\nprobe_wm.py writes them; "
                 f"on the cluster:  python v_1/src/world_models/probe_wm.py "
                 f"--method {method} --entity-type {ENTITY}")
    p = g[0]                       # one best-layer file per site; site order: any
    m = re.search(rf"{ENTITY}\.(\w+)\.layer(\d+)\.npz$", p)
    z = np.load(p)
    coef = np.asarray(z["coef"], np.float32).ravel()
    return coef, m.group(1), int(m.group(2)), os.path.basename(p)


def pairwise_eval(df, s, m, draws, seed):
    """E1's evaluation with a FIXED scorer: macro accuracy over ruler-pairs.
    No training -> the fold machinery is unnecessary; every drawn pair scores."""
    rp = P.eligible_ruler_pairs(df)
    accs = []
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        pairs = P.draw_pairs(df, m, rng, rp)
        pred = (s[pairs.pos_a.values] < s[pairs.pos_b.values])
        # scorer polarity: s is monotone in predicted year, so "a earlier" means
        # s_a < s_b. Sign errors show up as acc < .5, which is itself reported.
        correct = (pred.astype(int) == pairs.label.values).astype(float)
        t = pd.DataFrame({"c": correct,
                          "ra": np.minimum(pairs.ruler_a, pairs.ruler_b),
                          "rb": np.maximum(pairs.ruler_a, pairs.ruler_b)})
        accs.append(t.groupby(["ra", "rb"])["c"].mean().mean())
    return float(np.mean(accs)), float(np.std(accs))


def leace_erase(X, rulers):
    try:
        import torch
        from concept_erasure import LeaceEraser
    except ImportError:
        sys.exit("pip install concept-erasure (and torch) for the mediation test,"
                 " or pass --skip-leace")
    codes = pd.Categorical(rulers).codes
    Z = np.eye(codes.max() + 1, dtype=np.float32)[codes]
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    eraser = LeaceEraser.fit(Xt, torch.from_numpy(Z))
    return eraser(Xt).numpy()


def spearman(a, b):
    return float(stats.spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal", choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean",
                    help="fragment pooling site the direction is applied to")
    ap.add_argument("--dirs-root", default=DIRS_ROOT)
    ap.add_argument("--acts-root", default=None,
                    help="override the akkadian activation store (testing)")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--skip-leace", action="store_true")
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    if args.acts_root:
        PP.ACTS = args.acts_root
    coef, srcA_site, LA, src = find_cellA_direction(args.method, args.dirs_root)
    df = P.load_eligible()
    layers = PP.load_act_layers(args.method, args.variant, args.site, stride=1)
    year = df.year.values.astype(float)
    print(f"[dir] {src}: cell-A best layer {LA} (site {srcA_site}), "
          f"d={len(coef)} | fragment layers on disk: {len(layers)}", flush=True)

    def readout(X, tag):
        s = X @ coef
        rho = spearman(s, year)
        mac, sd = pairwise_eval(df, s, args.m, args.draws, args.seed)
        print(f"  [{tag}] spearman={rho:+.3f}  pairwise macro={mac:.3f}±{sd:.3f}",
              flush=True)
        return {"spearman": rho, "pairwise_macro": mac, "pairwise_sd": sd}

    out = {"method": args.method, "variant": args.variant, "site": args.site,
           "cellA_direction": src, "cellA_layer": LA, "entity_set": ENTITY,
           "m": args.m, "draws": args.draws, "n_fragments": int(len(df))}

    # primary: the same residual depth the direction was fitted at
    if LA not in layers:
        near = min(layers, key=lambda L: abs(L - LA))
        print(f"[warn] fragment layer {LA} not on disk; using nearest {near}",
              flush=True)
        LA = near
        out["cellA_layer_used"] = LA
    X = layers[LA]
    out["frozen"] = readout(X, f"frozen L{LA}")

    if not args.skip_leace:
        Xe = leace_erase(X, df.ruler.values)
        out["frozen_after_leace_ruler"] = readout(Xe, f"LEACE(ruler) L{LA}")
        # surgical control: erasure should barely move the vectors
        delta = float(np.linalg.norm(Xe - X) / np.linalg.norm(X))
        out["leace_relative_change"] = delta
        print(f"  [leace] relative representation change {delta:.4f} "
              "(rank<=39 nick out of d=%d)" % X.shape[1], flush=True)

    # exploratory: the frozen direction against every fragment layer
    out["layer_sweep"] = {int(L): spearman(Xl @ coef, year)
                          for L, Xl in layers.items()}

    # cosine vs the E1 pairwise direction (trained on order only), in the
    # standardized coordinates the pairwise probe actually lived in
    cos = {}
    for p in sorted(glob.glob(os.path.join(
            _PAIRS, "results", "directions",
            f"{args.method}.{args.variant}.{args.site}*.npz"))):
        mL = re.search(r"layer(-?\d+)\.npz$", p)
        Lp = int(mL.group(1))
        if Lp not in layers:
            continue
        w_pair = np.load(p)["w"].astype(np.float32)
        sd_feat = layers[Lp].std(axis=0) + 1e-8
        wA = coef * sd_feat            # cell-A direction in scaled coords
        c = float(wA @ w_pair / (np.linalg.norm(wA) * np.linalg.norm(w_pair)))
        cos[os.path.basename(p)] = {
            "cosine": c, "pairwise_layer": Lp, "cellA_layer": LA,
            "cross_layer": Lp != LA}
        print(f"  [cosine] vs {os.path.basename(p)}: {c:+.3f}"
              f"{'  (cross-layer)' if Lp != LA else ''}", flush=True)
    out["cosine_vs_pairwise_direction"] = cos

    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS,
                       f"{args.method}.{args.variant}.{args.site}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
