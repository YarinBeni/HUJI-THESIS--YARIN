"""J7 — P2 Godey geography mirror (CPU).

Predict the find-spot coordinates (lat/lon) of each ORCC text from the model's
mean-pooled activations, across the model ladder (the scale study), using the
gazetteer built in J1. This is the thesis's POSITIVE CONTROL + explicit-carrier
contrast: toponyms are written in the text, so geography *should* decode even
where year (indirect) does not — proving the year-null isn't a broken pipeline.

Per method x cleaning x layer:
  * PLS + ridge for lat and lon separately, GroupKFold-by-SITE (held-out sites),
  * great-circle error (km) on combined predictions vs a centroid baseline,
  * region classification macro-F1 (PLS-DA).
Emits results/p2_geography__<method>.json. Uses on-disk mean activations only —
no GPU, no re-extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from pls_utils import fit_pls_groupkfold, l2_normalize          # noqa: E402
from metrics import great_circle_km                              # noqa: E402

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
    return (np.array(rows), np.array(lat), np.array(lon),
            np.array(region), np.array(site))


def gc_cv(X, lat, lon, sites, n_splits):
    """GroupKFold-by-site great-circle error: predict lat & lon with PLS(k=3 fixed),
    score great-circle km. Compares to centroid baseline."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    errs, base = [], []
    for tr, te in gkf.split(X, lat, sites):
        pls_la = PLSRegression(n_components=3).fit(X[tr], lat[tr])
        pls_lo = PLSRegression(n_components=3).fit(X[tr], lon[tr])
        pla = pls_la.predict(X[te]).ravel(); plo = pls_lo.predict(X[te]).ravel()
        errs.append(great_circle_km(lat[te], lon[te], pla, plo))
        base.append(great_circle_km(lat[te], lon[te],
                                    np.full(te.shape, lat[tr].mean()),
                                    np.full(te.shape, lon[tr].mean())))
    e = np.concatenate(errs); b = np.concatenate(base)
    return {"gc_km_mean": float(e.mean()), "gc_km_median": float(np.median(e)),
            "centroid_km_mean": float(b.mean()),
            "skill_vs_centroid": float(1 - e.mean() / b.mean())}


def run(args):
    rows, lat, lon, region, sites = load_geo_labels()
    n_sites = len(set(sites))
    n_splits = min(5, n_sites)
    print(f"[{args.method}] geo rows={len(rows)} sites={n_sites}")
    out = {"method": args.method, "n_rows": int(len(rows)), "n_sites": int(n_sites),
           "cleanings": {}}
    for cleaning in ["tier0", "maximal"]:
        d = find_acts_dir(args.method, cleaning, "mean")
        if d is None:
            out["cleanings"][cleaning] = {"missing": True}
            continue
        layers = available_layers(d)
        per_layer = {}
        for L in layers:
            X = load_layer(d, L)[rows]
            Xn = l2_normalize(X)
            # lat/lon recoverability (spearman of each via PLS) + great-circle
            lat_pls = fit_pls_groupkfold(Xn, lat, sites, n_components=3, n_splits=n_splits)
            lon_pls = fit_pls_groupkfold(Xn, lon, sites, n_components=3, n_splits=n_splits)
            per_layer[L] = {
                "lat_spearman": lat_pls["spearman_mean"],
                "lon_spearman": lon_pls["spearman_mean"],
                "geo": gc_cv(Xn, lat, lon, sites, n_splits),
            }
        out["cleanings"][cleaning] = {
            "n_layers": len(layers),
            "best_layer_by_skill": max(per_layer, key=lambda L: per_layer[L]["geo"]["skill_vs_centroid"]),
            "per_layer": per_layer,
        }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p2_geography__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        if blk.get("missing"):
            continue
        bl = blk["best_layer_by_skill"]
        g = blk["per_layer"][bl]["geo"]
        print(f"  {cl:7s} best L{bl}: gc={g['gc_km_mean']:.0f}km "
              f"skill_vs_centroid={g['skill_vs_centroid']:+.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
