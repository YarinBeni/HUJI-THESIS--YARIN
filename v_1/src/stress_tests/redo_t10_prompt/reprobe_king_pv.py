"""J3 reprobe — year PLS + ridge on the prompted-king activations, per
(variant, layer, pool). GroupKFold-by-ruler; king_* pools drop NaN (name-not-found)
rows. Emits one summary JSON per model.

Usage:
    python reprobe_king_pv.py --acts-dir <out_dir>/prompted_king --model qwen3_8b \
        --out v_1/src/stress_tests/redo_t10_prompt/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing"))
from pls_utils import fit_pls_groupkfold, fit_ridge_year_groupkfold, l2_normalize  # noqa: E402

POOLS = ["mean", "king_last", "king_mean"]
PLS_KS = [1, 2, 3, 5]


def probe_one(X, years, rulers):
    """Return best-k PLS block + ridge block for one (layer,pool)."""
    mask = np.isfinite(X).all(axis=1)
    X, y, g = X[mask], years[mask].astype(float), rulers[mask]
    out = {"n": int(mask.sum()), "n_rulers": int(len(set(g)))}
    if out["n"] < 20 or out["n_rulers"] < 3:
        out["skipped"] = "insufficient coverage"
        return out
    Xn = l2_normalize(X)
    pls = {k: fit_pls_groupkfold(Xn, y, g, n_components=k, n_splits=min(5, out["n_rulers"]))
           for k in PLS_KS if k < out["n"]}
    best_k = max(pls, key=lambda k: pls[k]["spearman_mean"] if not np.isnan(pls[k]["spearman_mean"]) else -9)
    out["pls_best_k"] = best_k
    out["pls"] = pls[best_k]
    try:
        out["ridge"] = fit_ridge_year_groupkfold(Xn, y, np.log(y), g,
                                                  n_splits=min(5, out["n_rulers"]))
    except Exception as e:
        out["ridge"] = {"error": str(e)}
    return out


def run(args):
    acts_dir = Path(args.acts_dir)
    summary = {"model": args.model, "variants": {}}
    for vdir in sorted(acts_dir.glob("pv*")):
        variant = vdir.name
        vres = {}
        for npz in sorted(vdir.glob("L*.npz")):
            L = int(npz.stem[1:])
            d = np.load(npz, allow_pickle=True)
            years = d["years"]; rulers = d["rulers"]
            vres[L] = {pool: probe_one(d[pool], years, rulers) for pool in POOLS}
        summary["variants"][variant] = vres
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{args.model}__t10_king_summary.json"
    fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # console headline: best year spearman per (variant,pool)
    for v, vres in summary["variants"].items():
        for pool in POOLS:
            best = max((vres[L][pool].get("pls", {}).get("spearman_mean", float("nan"))
                        for L in vres), default=float("nan"))
            print(f"{args.model} {v:4s} {pool:9s} best-year-spearman={best:.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
