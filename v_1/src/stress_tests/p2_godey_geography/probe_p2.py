"""J7 — P2 Godey geography mirror (CPU).

Predict find-spot lat/lon from mean-pooled activations across the model ladder,
GroupKFold-by-SITE (held-out sites). Positive control + explicit-carrier contrast
(toponyms are in-text). Now sweeps PLS k in {1,2,3,5} (reports best-k), adds a
Ridge arm, and picks the best layer by great-circle skill vs a centroid baseline.
Emits results/p2_geography__<method>.json.
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
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from pls_utils import fit_pls_groupkfold, fit_ridge_year_groupkfold, l2_normalize  # noqa: E402
from metrics import great_circle_km                                  # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
GAZ = Path(__file__).resolve().parents[1] / "shared" / "sites_gazetteer.csv"
PLS_KS = [1, 2, 3, 5]


def load_geo_labels():
    df = pd.read_parquet(CORPUS)
    gaz = pd.read_csv(GAZ).dropna(subset=["lat", "lon"])
    gmap = {str(r.provenance): (r.lat, r.lon, r.region) for r in gaz.itertuples(index=False)}
    prov = [str(x) for x in df["provenance"]]
    rows, lat, lon, region, site = [], [], [], [], []
    for i, p in enumerate(prov):
        if p in gmap:
            rows.append(i); la, lo, rg = gmap[p]
            lat.append(la); lon.append(lo); region.append(str(rg)); site.append(p)
    return (np.array(rows), np.array(lat), np.array(lon), np.array(region), np.array(site))


def _best_pls_spearman(Xn, y, groups, n_splits):
    best = (-9.0, None)
    for k in PLS_KS:
        try:
            r = fit_pls_groupkfold(Xn, y, groups, n_components=k, n_splits=n_splits)
        except Exception:
            continue
        s = r["spearman_mean"]
        if s == s and s > best[0]:
            best = (s, k)
    return best  # (spearman, best_k)


def _ridge_spearman(Xn, y, groups, n_splits):
    try:
        return fit_ridge_year_groupkfold(Xn, y, np.log(y), groups, n_splits=n_splits)["raw"]["spearman_mean"]
    except Exception:
        return float("nan")


def gc_cv(X, lat, lon, sites, n_splits):
    """GroupKFold-by-site great-circle error, swept over k; returns best-k block."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    splits = list(GroupKFold(n_splits=n_splits).split(X, lat, sites))
    best = None
    for k in PLS_KS:
        errs, base, ok = [], [], True
        for tr, te in splits:
            try:
                pla = PLSRegression(n_components=k).fit(X[tr], lat[tr]).predict(X[te]).ravel()
                plo = PLSRegression(n_components=k).fit(X[tr], lon[tr]).predict(X[te]).ravel()
            except Exception:
                ok = False; break
            errs.append(great_circle_km(lat[te], lon[te], pla, plo))
            base.append(great_circle_km(lat[te], lon[te],
                                        np.full(te.shape, lat[tr].mean()),
                                        np.full(te.shape, lon[tr].mean())))
        if not ok:
            continue
        e = np.concatenate(errs); b = np.concatenate(base)
        res = {"k": k, "gc_km_mean": float(e.mean()), "gc_km_median": float(np.median(e)),
               "centroid_km_mean": float(b.mean()), "skill_vs_centroid": float(1 - e.mean() / b.mean())}
        if best is None or res["skill_vs_centroid"] > best["skill_vs_centroid"]:
            best = res
    return best or {"skipped": True}


def run(args):
    rows, lat, lon, region, sites = load_geo_labels()
    n_sites = len(set(sites)); n_splits = min(5, n_sites)
    print(f"[{args.method}] geo rows={len(rows)} sites={n_sites}")
    out = {"method": args.method, "protocol": "geo_groupkfold_by_site",
           "n_rows": int(len(rows)), "n_sites": int(n_sites), "cleanings": {}}
    for cleaning in ["tier0", "maximal"]:
        d = find_acts_dir(args.method, cleaning, "mean")
        if d is None:
            out["cleanings"][cleaning] = {"missing": True}; continue
        layers = available_layers(d); per_layer = {}
        for L in layers:
            Xn = l2_normalize(load_layer(d, L)[rows])
            lat_sp, lat_k = _best_pls_spearman(Xn, lat, sites, n_splits)
            lon_sp, lon_k = _best_pls_spearman(Xn, lon, sites, n_splits)
            per_layer[str(L)] = {
                "lat_spearman": lat_sp, "lat_best_k": lat_k,
                "lon_spearman": lon_sp, "lon_best_k": lon_k,
                "lat_ridge_spearman": _ridge_spearman(Xn, lat, sites, n_splits),
                "lon_ridge_spearman": _ridge_spearman(Xn, lon, sites, n_splits),
                "geo": gc_cv(Xn, lat, lon, sites, n_splits),
            }
        bestL = max(per_layer, key=lambda L: per_layer[L]["geo"].get("skill_vs_centroid", -9))
        out["cleanings"][cleaning] = {"n_layers": len(layers),
                                      "best_layer_by_skill": int(bestL),
                                      "per_layer": per_layer}
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p2_geography__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        if blk.get("missing"):
            continue
        bl = str(blk["best_layer_by_skill"]); pl = blk["per_layer"][bl]; g = pl["geo"]
        print(f"  {cl:7s} bestL{bl}: gc={g['gc_km_mean']:.0f}km(k={g.get('k')}) skill={g['skill_vs_centroid']:+.3f} "
              f"| lat PLS(k{pl['lat_best_k']})={pl['lat_spearman']:.3f} ridge={pl['lat_ridge_spearman']:.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
