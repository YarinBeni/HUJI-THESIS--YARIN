"""TF-IDF control for the tier0 baseline slide — YEAR + GEO balanced-MC probes
on TF-IDF features (char_wb(2,5) -> SVD-512, the suite's cited baseline
config), for the four text variants tier0 / maximal / engtier0 / engmaximal.

Same protocols as the model probes: YEAR = 8-ruler balanced MC + GroupKFold-
by-ruler (shared/mc_probe.mc_year_probe); GEO = 10-merged-site balanced MC +
GroupKFold-by-site (p2_godey_geography/probe_p2_mc.mc_layer). CPU, minutes.

Usage:  python run_tfidf_baseline.py
Writes  results/tfidf_baseline.json + results/csv/tfidf_baseline.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_THIS = Path(__file__).resolve()
_ST = _THIS.parents[1]
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_ST / "shared"))
sys.path.insert(0, str(_ST / "p2_godey_geography"))
from mc_probe import mc_year_probe                      # noqa: E402
from probe_p2_mc import mc_layer                        # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
SITES = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset_sites"
TRANSLATIONS = _ST / "translation/translations.parquet"
SVD_DIM = 512
CLEANINGS = ["tier0", "maximal", "engtier0", "engmaximal"]


def texts_for(df, cleaning):
    if cleaning in ("tier0", "maximal"):
        return df[f"text_{cleaning}"].fillna("").astype(str).tolist()
    col = "eng_tier0" if cleaning == "engtier0" else "eng_maximal"
    tr = pd.read_parquet(TRANSLATIONS).set_index("fragment_id")[col]
    return tr.reindex(df["fragment_id"].astype(str)).fillna("").astype(str).tolist()


def tfidf_svd(texts):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    Xs = normalize(vec.fit_transform(texts), norm="l2")
    svd = TruncatedSVD(n_components=min(SVD_DIM, Xs.shape[1] - 1), random_state=0)
    return svd.fit_transform(Xs).astype(np.float32)


def main():
    df = pd.read_parquet(PARQUET)
    fids = json.load(open(BAL / "corpus_fragment_order.json"))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    year_rows = [np.where(r)[0] for r in np.load(BAL / "draws_matrix.npy")]

    site_draws = np.load(SITES / "draws_matrix.npy")
    labels = json.loads((SITES / "site_labels.json").read_text())
    manifest = json.loads((SITES / "manifest.json").read_text())
    site_lab = np.array([x if x is not None else "" for x in labels])
    coord = {s: (v["lat"], v["lon"]) for s, v in manifest["sites"].items()}
    lat = np.array([coord.get(s, (np.nan, np.nan))[0] for s in site_lab])
    lon = np.array([coord.get(s, (np.nan, np.nan))[1] for s in site_lab])
    geo_rows = [np.where(site_draws[d])[0] for d in range(site_draws.shape[0])]

    out = {"method": "tfidf", "protocol": "tier0-baseline controls",
           "tfidf": "char_wb(2,5), SVD-512 (baseline convention)",
           "cleanings": {}}
    rows = []
    for cl in CLEANINGS:
        t0 = time.time()
        X = tfidf_svd(texts_for(df, cl))
        yr = mc_year_probe(X, year, ruler, year_rows)
        geo = mc_layer(X, lat, lon, site_lab, geo_rows)
        out["cleanings"][cl] = {"year": yr, "geo": geo}
        rows.append([cl, round(yr["spearman_mean"], 4), round(yr["spearman_std"], 4),
                     yr.get("best_k"), round(yr["ridge"]["spearman_mean"], 4),
                     round(geo["gc_km_mean"], 1), geo.get("best_k"),
                     round(geo["ridge"]["gc_km_mean"], 1)])
        print(f"[{cl}] year PLS {yr['spearman_mean']:.3f}+-{yr['spearman_std']:.3f} "
              f"(k={yr.get('best_k')})  ridge {yr['ridge']['spearman_mean']:.3f}  |  "
              f"geo PLS {geo['gc_km_mean']:.0f} km (k={geo.get('best_k')})  "
              f"ridge {geo['ridge']['gc_km_mean']:.0f} km   ({time.time()-t0:.0f}s)",
              flush=True)

    (_ST / "results/tfidf_baseline.json").write_text(json.dumps(out, indent=2))
    with open(_ST / "results/csv/tfidf_baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cleaning", "year_pls_spearman", "year_pls_std", "year_best_k",
                    "year_ridge_spearman", "geo_pls_km", "geo_best_k", "geo_ridge_km"])
        w.writerows(rows)
    print("wrote results/tfidf_baseline.json + results/csv/tfidf_baseline.csv")


if __name__ == "__main__":
    main()
