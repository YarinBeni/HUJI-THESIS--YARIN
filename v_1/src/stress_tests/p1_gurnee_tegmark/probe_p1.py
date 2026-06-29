"""J6 — P1 Gurnee-Tegmark year-probe (CPU).

Year recovery from frozen activations at the three sites, GroupKFold-by-ruler:
  * mean       — on-disk <method>_{tier0,maximal}_mean (all layers)
  * king_last  — <method>_tier0_kinglast  (from J4; drop name-not-found rows)
  * king_mean  — <method>_tier0_kingmean
Adds Gurnee-Tegmark's linearity check (1-hidden-layer MLP vs linear PLS at the
best layer) and proximity_error. Emits results/p1_year__<method>.json.

The contrast that matters: year recoverable at king_* (where the model knows the
fact) but not at mean -> local-but-not-global; at neither -> hard dissociation;
grouped << random-split (in pls_utils shuffled baseline) -> ruler-memorization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from geo_loader import find_acts_dir, load_layer, available_layers   # noqa: E402
from pls_utils import fit_pls_groupkfold, fit_ridge_year_groupkfold, l2_normalize  # noqa: E402
from metrics import proximity_error                              # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"
PLS_KS = [1, 2, 3, 5]


def _year_groups():
    df = pd.read_parquet(CORPUS)
    mask = df["year"].notna().to_numpy()
    return mask, df["year"].to_numpy(), df["ruler"].astype(str).to_numpy()


def _probe_block(X, y, g, n_splits=5):
    Xn = l2_normalize(X)
    pls = {k: fit_pls_groupkfold(Xn, y, g, n_components=k, n_splits=n_splits)
           for k in PLS_KS if k < len(y)}
    bk = max(pls, key=lambda k: (pls[k]["spearman_mean"]
                                 if not np.isnan(pls[k]["spearman_mean"]) else -9))
    blk = {"pls_best_k": bk, "pls": pls[bk]}
    try:
        blk["ridge"] = fit_ridge_year_groupkfold(Xn, y, np.log(y), g, n_splits=n_splits)
    except Exception as e:
        blk["ridge"] = {"error": str(e)}
    return blk, Xn


def _linearity_check(Xn, y, g, n_splits=5):
    """MLP(1 hidden) vs linear PLS spearman at this layer. MLP~=linear => the
    info is (not) linearly accessible, not a capacity problem."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr
    gkf = GroupKFold(n_splits=n_splits)
    sp = []
    for tr, te in gkf.split(Xn, y, g):
        if len(set(y[te])) < 2:
            continue
        mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=42)
        mlp.fit(Xn[tr], y[tr])
        sp.append(spearmanr(y[te], mlp.predict(Xn[te])).statistic)
    return float(np.nanmean(sp)) if sp else float("nan")


def _mean_site(method, mask, y, g, out):
    for cleaning in ["tier0", "maximal"]:
        d = find_acts_dir(method, cleaning, "mean")
        if d is None:
            out[f"mean_{cleaning}"] = {"missing": True}; continue
        layers = available_layers(d)
        per = {}
        for L in layers:
            X = load_layer(d, L)[mask]
            blk, _ = _probe_block(X, y, g)
            per[L] = blk
        bestL = max(per, key=lambda L: per[L]["pls"]["spearman_mean"]
                    if not np.isnan(per[L]["pls"]["spearman_mean"]) else -9)
        Xb = l2_normalize(load_layer(d, bestL)[mask])
        out[f"mean_{cleaning}"] = {
            "n": int(mask.sum()), "best_layer": bestL,
            "best_spearman": per[bestL]["pls"]["spearman_mean"],
            "mlp_spearman_at_best": _linearity_check(Xb, y, g),
            "per_layer": per,
        }


def _king_site(method, pool_dir, label, full_y, full_ruler, out):
    d = ACTS / f"{method}_tier0_{pool_dir}"
    cov_fp = d / "king_coverage.json"
    if not cov_fp.exists() or not any(d.glob("layer_*.npz")):
        out[label] = {"missing": True}; return
    cov = json.loads(cov_fp.read_text())
    found = np.array(cov["found"], dtype=bool)
    yr = np.array(cov["years"], dtype=float)
    rl = np.array(cov["rulers"])
    ymask = found & np.isfinite(yr)
    layers = available_layers(d)
    per = {}
    for L in layers:
        X = load_layer(d, L)            # full corpus order, NaN where not found
        rowmask = ymask & np.isfinite(X).all(axis=1)
        if rowmask.sum() < 20 or len(set(rl[rowmask])) < 3:
            continue
        blk, _ = _probe_block(X[rowmask], yr[rowmask], rl[rowmask],
                              n_splits=min(5, len(set(rl[rowmask]))))
        per[L] = blk
    if not per:
        out[label] = {"insufficient_coverage": True, "n_found": int(found.sum())}; return
    bestL = max(per, key=lambda L: per[L]["pls"]["spearman_mean"]
                if not np.isnan(per[L]["pls"]["spearman_mean"]) else -9)
    out[label] = {"n_found": int(found.sum()), "coverage": round(float(found.mean()), 3),
                  "best_layer": bestL, "best_spearman": per[bestL]["pls"]["spearman_mean"],
                  "per_layer": per}


def run(args):
    mask, y_all, ruler_all = _year_groups()
    y = y_all[mask]; g = ruler_all[mask]
    out = {"method": args.method}
    _mean_site(args.method, mask, y, g, out)
    _king_site(args.method, "kinglast", "king_last", y_all, ruler_all, out)
    _king_site(args.method, "kingmean", "king_mean", y_all, ruler_all, out)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p1_year__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for site in ["mean_tier0", "mean_maximal", "king_last", "king_mean"]:
        s = out.get(site, {})
        if s.get("missing") or s.get("insufficient_coverage"):
            print(f"  {site:13s} -> {'missing' if s.get('missing') else 'insufficient coverage'}")
        else:
            print(f"  {site:13s} best L{s['best_layer']} spearman={s['best_spearman']:.3f}"
                  + (f" (cov {s['coverage']})" if "coverage" in s else
                     f"  MLP={s.get('mlp_spearman_at_best', float('nan')):.3f}"))
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
