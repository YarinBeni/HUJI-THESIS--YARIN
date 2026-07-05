"""J14 — P2 geography under SITE-BALANCED Monte-Carlo draws (CPU).

The site-side mirror of the P1 balanced-MC protocol: 200 balanced draws (10 merged
sites x k=21, built by build_site_balanced_subset.py), GroupKFold-by-MERGED-SITE
within each draw (held-out find-spots), PLS swept over k in {1,2,3,5} + a Ridge arm
for lat & lon, plus the great-circle error / skill-vs-centroid decoding view.
Averaged over draws (mean +- std). Runs on both cleanings (tier0 + maximal) from
the on-disk mean activations. Emits results/mc/p2_geo_mc__<method>.json.

Usage:  python probe_p2_mc.py --method qwen3_8b [--n-jobs 16]
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
from pls_utils import l2_normalize                                   # noqa: E402
from metrics import great_circle_km                                   # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
SUBSET = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset_sites"
PLS_KS = [1, 2, 3, 5]


def _mean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def _std(v):
    v = [x for x in v if x == x]
    return float(np.std(v)) if v else float("nan")


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(np.asarray(a).tolist())) < 2:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def one_draw(Xn, lat, lon, site, ks, n_splits=5):
    """One balanced draw: GroupKFold-by-site. Returns per-k gc/skill + lat/lon
    Spearman, and a ridge arm."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    ns = min(n_splits, len(set(site.tolist())))
    if ns < 2:
        return None
    splits = list(GroupKFold(n_splits=ns).split(Xn, lat, site))
    out = {}
    for k in ks:
        if k >= len(Xn):
            continue
        gc, base, lsp, osp = [], [], [], []
        try:
            for tr, te in splits:
                pla = PLSRegression(n_components=k).fit(Xn[tr], lat[tr]).predict(Xn[te]).ravel()
                plo = PLSRegression(n_components=k).fit(Xn[tr], lon[tr]).predict(Xn[te]).ravel()
                gc.append(great_circle_km(lat[te], lon[te], pla, plo))
                base.append(great_circle_km(lat[te], lon[te],
                                            np.full(te.shape, lat[tr].mean()),
                                            np.full(te.shape, lon[tr].mean())))
                lsp.append(_spearman(lat[te], pla)); osp.append(_spearman(lon[te], plo))
        except Exception:
            continue
        e, b = np.concatenate(gc), np.concatenate(base)
        out[k] = {"gc_km": float(e.mean()), "skill": float(1 - e.mean() / b.mean()),
                  "lat_sp": _mean(lsp), "lon_sp": _mean(osp)}
    ridge = {}
    try:
        gc, base, lsp, osp = [], [], [], []
        for tr, te in splits:
            pla = Ridge(alpha=10.0).fit(Xn[tr], lat[tr]).predict(Xn[te])
            plo = Ridge(alpha=10.0).fit(Xn[tr], lon[tr]).predict(Xn[te])
            gc.append(great_circle_km(lat[te], lon[te], pla, plo))
            base.append(great_circle_km(lat[te], lon[te],
                                        np.full(te.shape, lat[tr].mean()),
                                        np.full(te.shape, lon[tr].mean())))
            lsp.append(_spearman(lat[te], pla)); osp.append(_spearman(lon[te], plo))
        e, b = np.concatenate(gc), np.concatenate(base)
        ridge = {"gc_km": float(e.mean()), "skill": float(1 - e.mean() / b.mean()),
                 "lat_sp": _mean(lsp), "lon_sp": _mean(osp)}
    except Exception:
        pass
    return out, ridge


def mc_layer(X, lat, lon, site_lab, draw_rows, ks=PLS_KS):
    per_k = {k: {"gc": [], "sk": [], "lat": [], "lon": []} for k in ks}
    rid = {"gc": [], "sk": [], "lat": [], "lon": []}
    used = 0
    for rows in draw_rows:
        Xs = X[rows]
        m = np.isfinite(Xs).all(axis=1)
        rows = rows[m]
        if len(rows) < 20:
            continue
        res = one_draw(l2_normalize(X[rows]), lat[rows], lon[rows], site_lab[rows], ks)
        if res is None:
            continue
        perk, ridge = res
        for k, v in perk.items():
            per_k[k]["gc"].append(v["gc_km"]); per_k[k]["sk"].append(v["skill"])
            per_k[k]["lat"].append(v["lat_sp"]); per_k[k]["lon"].append(v["lon_sp"])
        if ridge:
            rid["gc"].append(ridge["gc_km"]); rid["sk"].append(ridge["skill"])
            rid["lat"].append(ridge["lat_sp"]); rid["lon"].append(ridge["lon_sp"])
        used += 1
    if used == 0:
        return {"skipped": True}
    pk = {str(k): {"gc_km_mean": _mean(per_k[k]["gc"]), "skill_mean": _mean(per_k[k]["sk"]),
                   "skill_std": _std(per_k[k]["sk"]), "lat_spearman": _mean(per_k[k]["lat"]),
                   "lon_spearman": _mean(per_k[k]["lon"])} for k in ks}
    best = max(pk, key=lambda kk: pk[kk]["skill_mean"] if pk[kk]["skill_mean"] == pk[kk]["skill_mean"] else -9)
    return {"n_draws_used": used, "best_k": int(best), "per_k": pk, **pk[best],
            "ridge": {"gc_km_mean": _mean(rid["gc"]), "skill_mean": _mean(rid["sk"]),
                      "lat_spearman": _mean(rid["lat"]), "lon_spearman": _mean(rid["lon"])}}


def run(args):
    df = pd.read_parquet(args.corpus)
    draws = np.load(Path(args.subset) / "draws_matrix.npy")
    labels = json.loads((Path(args.subset) / "site_labels.json").read_text())
    manifest = json.loads((Path(args.subset) / "manifest.json").read_text())
    site_lab = np.array([x if x is not None else "" for x in labels])
    coord = {s: (v["lat"], v["lon"]) for s, v in manifest["sites"].items()}
    lat = np.array([coord.get(s, (np.nan, np.nan))[0] for s in site_lab])
    lon = np.array([coord.get(s, (np.nan, np.nan))[1] for s in site_lab])
    draw_rows = [np.where(draws[d])[0] for d in range(draws.shape[0])]

    out = {"method": args.method, "protocol": "mc_balanced_sites",
           "n_draws": int(draws.shape[0]), "k": manifest["k"],
           "n_sites": manifest["n_sites"], "cleanings": {}}
    from joblib import Parallel, delayed
    for cleaning in ["tier0", "maximal"]:
        d = find_acts_dir(args.method, cleaning, "mean")
        if d is None:
            out["cleanings"][cleaning] = {"missing": True}; continue
        layers = available_layers(d)

        def one(L):
            return L, mc_layer(load_layer(d, L), lat, lon, site_lab, draw_rows)
        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(L) for L in layers))
        per = {str(L): res[L] for L in layers}
        valid = {L: r for L, r in res.items() if not r.get("skipped")}
        if not valid:
            out["cleanings"][cleaning] = {"insufficient": True, "per_layer": per}; continue
        bestL = max(valid, key=lambda L: valid[L]["skill_mean"])
        out["cleanings"][cleaning] = {"best_layer": bestL, "best": valid[bestL], "per_layer": per}

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p2_geo_mc__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        if blk.get("missing") or blk.get("insufficient"):
            print(f"  {cl:8s}: {'missing' if blk.get('missing') else 'insufficient'}"); continue
        b = blk["best"]
        print(f"  {cl:8s}: L{blk['best_layer']} gc={b['gc_km_mean']:.0f}km "
              f"skill={b['skill_mean']:+.3f}±{b['skill_std']:.2f} (k={b['best_k']}) "
              f"lat_sp={b['lat_spearman']:.3f} lon_sp={b['lon_spearman']:.3f} "
              f"| ridge skill={b['ridge']['skill_mean']:+.3f} (draws={b['n_draws_used']})")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--subset", default=str(SUBSET))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "mc"))
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
