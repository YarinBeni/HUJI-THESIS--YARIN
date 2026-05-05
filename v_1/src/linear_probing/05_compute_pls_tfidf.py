# Run locally: python v_1/src/linear_probing/05_compute_pls_tfidf.py

"""
TF-IDF PLS pipeline driver.

Refits TF-IDF (char_wb, ngram 2-5) on the union of SEAL+ORCC texts,
L2-normalizes rows, then runs the shared PLS pipeline for each
cleaning in {tier0, maximal}.  No GPU required; runtime < 10 min.
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

from pls_utils import fit_pls_groupkfold, fit_pls_full, project, l2_normalize  # noqa: E402

SEAL_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
ORCC_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_DIR      = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round1/pls"

TFIDF_PARAMS       = dict(analyzer="char_wb", ngram_range=(2, 5))
K_VALUES           = [1, 2, 3, 5]
N_SPLITS           = 5
N_COMPONENTS_FULL  = 5
CLEANINGS          = ["tier0", "maximal"]
YEAR_TRANSFORMS    = ["raw", "log"]


def main():
    print("Run locally — no GPU required. Estimated runtime < 10 min.", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquets...", flush=True)
    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)

    n_seal  = len(seal_df)   # 384
    n_orcc  = len(orcc_df)   # 1202
    n_total = n_seal + n_orcc  # 1586
    print(f"  SEAL: {n_seal}, ORCC: {n_orcc}, total: {n_total}", flush=True)

    # Fragment IDs: SEAL first then ORCC, matching parquet order
    fragment_ids = (
        seal_df["fragment_id"].astype(str).tolist()
        + orcc_df["fragment_id"].astype(str).tolist()
    )

    # Labeled ORCC rows (year not null)
    labeled_mask           = orcc_df["year"].notna()
    labeled_orcc_positions = np.where(labeled_mask)[0]   # within ORCC df
    labeled_idx            = labeled_orcc_positions + n_seal  # within 1586-row matrix

    y_raw  = orcc_df.loc[labeled_mask, "year"].values.astype(float)
    y_log  = np.log(y_raw)
    groups = orcc_df.loc[labeled_mask, "ruler"].astype(str).values

    n_labeled = len(labeled_idx)
    n_groups  = int(pd.Series(groups).nunique())
    print(f"  Labeled ORCC: {n_labeled}, unique rulers: {n_groups}", flush=True)

    all_results    = {}
    all_projections = {"fragment_ids": fragment_ids, "embeddings": {}}

    for cleaning in CLEANINGS:
        col = f"text_{cleaning}"
        print(f"\n=== TF-IDF ({cleaning}) ===", flush=True)

        all_texts = (
            seal_df[col].astype(str).tolist()
            + orcc_df[col].astype(str).tolist()
        )

        vec      = TfidfVectorizer(**TFIDF_PARAMS)
        X_sparse = vec.fit_transform(all_texts)   # (1586, V) sparse
        print(f"  Vocab size: {X_sparse.shape[1]}, matrix: {X_sparse.shape}", flush=True)

        # L2-normalize rows while still sparse (zeros unchanged, cheap)
        X_sparse = normalize(X_sparse, norm="l2")

        # Densify once — ~1586 × V × 4 bytes (V ≈ 50-100k → ~300-600 MB, OK)
        print("  Converting to dense...", flush=True)
        X = X_sparse.toarray().astype(np.float32)

        X_labeled = X[labeled_idx]   # (893, V)

        for year_transform in YEAR_TRANSFORMS:
            y          = y_log if year_transform == "log" else y_raw
            config_key = f"tfidf__{cleaning}__na__L00__year-{year_transform}"
            print(f"  {config_key}...", flush=True)

            metrics_per_k = fit_pls_groupkfold(
                X_labeled, y, groups,
                k_values=K_VALUES,
                n_splits=N_SPLITS,
            )

            best_k_by_spearman = max(
                K_VALUES, key=lambda k: metrics_per_k[str(k)]["spearman_mean"]
            )
            best_k_by_r2 = max(
                K_VALUES, key=lambda k: metrics_per_k[str(k)]["r2_mean"]
            )

            all_results[config_key] = {
                "method":           "tfidf",
                "cleaning":         cleaning,
                "pooling":          "na",
                "layer":            0,
                "year_transform":   year_transform,
                "n_labeled":        n_labeled,
                "n_groups":         n_groups,
                "metrics_per_k":    metrics_per_k,
                "best_k_by_spearman": best_k_by_spearman,
                "best_k_by_r2":     best_k_by_r2,
            }

            # Refit on full labeled set (n_components=5), project all 1586 rows
            pls_full = fit_pls_full(X_labeled, y, n_components=N_COMPONENTS_FULL)
            X_proj   = project(pls_full, X)   # (1586, 5)

            proj_base = f"tfidf__{cleaning}__L00"
            all_projections["embeddings"][f"{proj_base}__pls12-{year_transform}"] = (
                X_proj[:, [0, 1]].tolist()
            )
            all_projections["embeddings"][f"{proj_base}__pls23-{year_transform}"] = (
                X_proj[:, [1, 2]].tolist()
            )
            all_projections["embeddings"][f"{proj_base}__pls34-{year_transform}"] = (
                X_proj[:, [2, 3]].tolist()
            )

    results_path     = OUT_DIR / "pls_results_tfidf.json"
    projections_path = OUT_DIR / "pls_projections_tfidf.json"

    print(f"\nWriting {results_path}...", flush=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Writing {projections_path}...", flush=True)
    with open(projections_path, "w", encoding="utf-8") as f:
        json.dump(all_projections, f)

    print(f"\nDone. {len(all_results)} configs written.", flush=True)
    for key, res in all_results.items():
        bk = res["best_k_by_spearman"]
        sp = res["metrics_per_k"][str(bk)]["spearman_mean"]
        print(f"  {key}: best_k={bk}, spearman={sp:.4f}", flush=True)


if __name__ == "__main__":
    main()
