"""J17 step 3 — probe the English-translation embeddings (CPU).

SETUP   translations of tier0 + maximal (cleaning tags engtier0/engmaximal),
        mean pool, balanced Monte-Carlo:
          YEAR      — the P1 protocol: 8 rulers x 21 x 200 draws, GroupKFold-by-ruler
          GEOGRAPHY — the J14 protocol: 10 merged sites x 21 x 200 draws,
                      GroupKFold-by-merged-site (lat & lon separately)
PROBE   PLS (k swept over {1,2,3,5}, best-k) + Ridge
METRIC  Spearman(predicted, true) — year / lat / lon. Nothing else.

The comparison that matters: these rows vs the SAME model's Akkadian rows
(p1_year_mc / p2_geo_mc) — does translating to English surface the signal?

Usage:  python probe_translation_mc.py --method qwen3_8b
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_godey_geography"))
from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from mc_probe import mc_year_probe, draws_to_rows                    # noqa: E402
from probe_p2_mc import mc_layer as geo_mc_layer                     # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
SUB = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0"
CLEANINGS = ["engtier0", "engmaximal"]


def run(args):
    df = pd.read_parquet(CORPUS)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()

    ydraws = np.load(SUB / "balanced_subset/draws_matrix.npy")
    ydraw_rows = draws_to_rows(ydraws)

    sdraws = np.load(SUB / "balanced_subset_sites/draws_matrix.npy")
    slabels = json.loads((SUB / "balanced_subset_sites/site_labels.json").read_text())
    sman = json.loads((SUB / "balanced_subset_sites/manifest.json").read_text())
    site = np.array([x if x is not None else "" for x in slabels])
    coord = {s: (v["lat"], v["lon"]) for s, v in sman["sites"].items()}
    lat = np.array([coord.get(s, (np.nan, np.nan))[0] for s in site])
    lon = np.array([coord.get(s, (np.nan, np.nan))[1] for s in site])
    sdraw_rows = [np.where(sdraws[d])[0] for d in range(sdraws.shape[0])]

    from joblib import Parallel, delayed
    out = {"method": args.method, "protocol": "translation_mc",
           "metric": "spearman", "cleanings": {}}
    for cl in CLEANINGS:
        d = find_acts_dir(args.method, cl, "mean")
        if d is None:
            out["cleanings"][cl] = {"missing": True}; continue
        layers = available_layers(d)

        def one(L):
            X = load_layer(d, L)
            return L, {"year": mc_year_probe(X, year, ruler, ydraw_rows),
                       "geo": geo_mc_layer(X, lat, lon, site, sdraw_rows)}
        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(L) for L in layers))
        per = {str(L): res[L] for L in layers}
        vy = {L: r["year"] for L, r in res.items() if not r["year"].get("skipped")}
        vg = {L: r["geo"] for L, r in res.items() if not r["geo"].get("skipped")}
        blk = {"per_layer": per}
        if vy:
            bL = max(vy, key=lambda L: vy[L]["spearman_mean"])
            blk["year_best_layer"] = bL
            blk["year_best"] = {k: vy[bL][k] for k in
                                ("best_k", "spearman_mean", "spearman_std", "ridge")}
        if vg:
            bL = max(vg, key=lambda L: (vg[L]["lat_spearman"] + vg[L]["lon_spearman"]))
            blk["geo_best_layer"] = bL
            blk["geo_best"] = vg[bL]
        out["cleanings"][cl] = blk

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"trans_mc__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        if blk.get("missing"):
            print(f"  {cl:11s}: missing"); continue
        y = blk.get("year_best", {}); g = blk.get("geo_best", {})
        print(f"  {cl:11s}: year sp={y.get('spearman_mean', float('nan')):.3f}"
              f"±{y.get('spearman_std', float('nan')):.2f} "
              f"(ridge {y.get('ridge', {}).get('spearman_mean', float('nan')):.3f}) "
              f"| lat sp={g.get('lat_spearman', float('nan')):.3f} "
              f"lon sp={g.get('lon_spearman', float('nan')):.3f} "
              f"(ridge {g.get('ridge', {}).get('lat_spearman', float('nan')):.3f}/"
              f"{g.get('ridge', {}).get('lon_spearman', float('nan')):.3f})")
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--n-jobs", type=int, default=8)
    run(p.parse_args())
