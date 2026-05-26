# Run locally: python v_1/src/linear_probing/05_compute_ridge_tfidf.py

"""TF-IDF Ridge year-regression driver (cls_numeric probe, local).

Mirrors 05_compute_pls_tfidf.py's matrix build (char_wb 2-5 TF-IDF refit on the
SEAL+ORCC union, L2-normalized rows) but runs the Ridge year readout via
pls_utils.fit_ridge_year_groupkfold, so the imbalanced Ridge table (T2) has a tfidf
row alongside the qwen3 models. Writes records in the same config_key format as
round2_phase3/probe_thalesian.py so build_experiment_tables.py picks them up.
No GPU; runtime < 10 min.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pls_utils import fit_ridge_year_groupkfold  # noqa: E402

SEAL_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
ORCC_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_DIR      = REPO_ROOT / "v_1/src/linear_probing/results/orcc__probe_cls_numeric"

TFIDF_PARAMS    = dict(analyzer="char_wb", ngram_range=(2, 5))
N_SPLITS        = 5
RIDGE_ALPHA     = 1.0
CLEANINGS       = ["tier0", "maximal"]
YEAR_TRANSFORMS = ["raw", "log"]


def main():
    print("Run locally — no GPU required. Estimated runtime < 10 min.", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)
    n_seal = len(seal_df)
    print(f"  SEAL: {n_seal}, ORCC: {len(orcc_df)}", flush=True)

    labeled_mask = orcc_df["year"].notna()
    labeled_idx  = np.where(labeled_mask)[0] + n_seal     # within SEAL+ORCC matrix
    y_raw  = orcc_df.loc[labeled_mask, "year"].values.astype(float)
    y_log  = np.log(y_raw)
    groups = orcc_df.loc[labeled_mask, "ruler"].astype(str).values
    n_labeled = len(labeled_idx)
    n_groups  = int(pd.Series(groups).nunique())
    print(f"  Labeled ORCC: {n_labeled}, unique rulers: {n_groups}", flush=True)

    results_path = OUT_DIR / "cls_numeric_results_tfidf.json"
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for cleaning in CLEANINGS:
        col = f"text_{cleaning}"
        print(f"\n=== TF-IDF Ridge ({cleaning}) ===", flush=True)
        all_texts = seal_df[col].astype(str).tolist() + orcc_df[col].astype(str).tolist()
        X = normalize(TfidfVectorizer(**TFIDF_PARAMS).fit_transform(all_texts), norm="l2")
        X = X.toarray().astype(np.float32)
        X_labeled = X[labeled_idx]

        ridge_results = fit_ridge_year_groupkfold(
            X_labeled, y_raw, y_log, groups, n_splits=N_SPLITS, alpha=RIDGE_ALPHA)
        for yt in YEAR_TRANSFORMS:
            r = ridge_results[yt]
            cfg_key = f"tfidf__{cleaning}__na__L00__year-{yt}"
            all_results[cfg_key] = {
                "method": "tfidf", "cleaning": cleaning, "pooling": "na",
                "layer": 0, "probe": "ridge", "year_transform": yt,
                "n_labeled": n_labeled, "n_groups": n_groups, "alpha": RIDGE_ALPHA,
                **r,
            }
            print(f"  {cfg_key}  sp={r['spearman_mean']:.3f}  "
                  f"mae={r['mae_mean']:.0f}  r2={r['r2_mean']:.3f}", flush=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone → {results_path}  ({len(all_results)} keys)", flush=True)


if __name__ == "__main__":
    main()
