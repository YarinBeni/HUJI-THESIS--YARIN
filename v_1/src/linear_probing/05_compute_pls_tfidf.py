# Run locally: python v_1/src/linear_probing/05_compute_pls_tfidf.py

"""
TF-IDF PLS pipeline driver.

Refits TF-IDF (char_wb, ngram 2-5) on the union of SEAL+ORCC texts,
L2-normalizes rows, then runs the shared PLS pipeline for each
cleaning in {tier0, maximal}.  No GPU required; runtime < 10 min.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pls_utils import (  # noqa: E402
    fit_pls_groupkfold, fit_pls_full,
    fit_plsda_stratified_kfold, fit_plsda_full,
    project, l2_normalize,
)

SEAL_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
ORCC_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_DIR      = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round1/pls"

TFIDF_PARAMS       = dict(analyzer="char_wb", ngram_range=(2, 5))
K_VALUES           = [1, 2, 3, 5]
N_SPLITS           = 5
N_COMPONENTS_FULL  = 5
CLEANINGS          = ["tier0", "maximal"]
YEAR_TRANSFORMS    = ["raw", "log"]


def parse_args():
    p = argparse.ArgumentParser(description="TF-IDF PLS pipeline (local)")
    p.add_argument("--target", default="both", choices=["year", "ruler", "both"])
    p.add_argument("--overwrite", action="store_true",
                   help="Clear existing tfidf keys before writing")
    return p.parse_args()


def main():
    args = parse_args()
    targets = ["year", "ruler"] if args.target == "both" else [args.target]

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

    y_ruler = orcc_df.loc[labeled_mask, "ruler"].astype(str).values

    n_labeled = len(labeled_idx)
    n_groups  = int(pd.Series(groups).nunique())
    print(f"  Labeled ORCC: {n_labeled}, unique rulers: {n_groups}", flush=True)

    # Load existing results for merge
    results_path     = OUT_DIR / "pls_results_tfidf.json"
    projections_path = OUT_DIR / "pls_projections_tfidf.json"

    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if projections_path.exists():
        with open(projections_path) as f:
            existing_proj = json.load(f)
        all_projections = {"fragment_ids": fragment_ids,
                           "embeddings": dict(existing_proj.get("embeddings", {}))}
    else:
        all_projections = {"fragment_ids": fragment_ids, "embeddings": {}}

    if args.overwrite:
        cleared = [k for k in list(all_results) if k.startswith("tfidf__")]
        for k in cleared:
            del all_results[k]
        if cleared:
            print(f"  [overwrite] Cleared {len(cleared)} existing tfidf keys")

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
        proj_base = f"tfidf__{cleaning}__L00"

        if "year" in targets:
            for year_transform in YEAR_TRANSFORMS:
                y          = y_log if year_transform == "log" else y_raw
                config_key = f"tfidf__{cleaning}__na__L00__year-{year_transform}"
                print(f"  {config_key}...", flush=True)

                metrics_per_k = {}
                for k in K_VALUES:
                    print(f"    k={k}...", flush=True)
                    metrics_per_k[str(k)] = fit_pls_groupkfold(
                        X_labeled, y, groups, n_components=k, n_splits=N_SPLITS)

                best_k_by_spearman = max(K_VALUES, key=lambda k: metrics_per_k[str(k)]["spearman_mean"])
                best_k_by_r2       = max(K_VALUES, key=lambda k: metrics_per_k[str(k)]["r2_mean"])

                all_results[config_key] = {
                    "method": "tfidf", "cleaning": cleaning, "pooling": "na",
                    "layer": 0, "year_transform": year_transform,
                    "n_labeled": n_labeled, "n_groups": n_groups,
                    "metrics_per_k": metrics_per_k,
                    "best_k_by_spearman": best_k_by_spearman,
                    "best_k_by_r2": best_k_by_r2,
                }

                pls_full = fit_pls_full(X_labeled, y, n_components=N_COMPONENTS_FULL)
                X_proj   = project(pls_full, X)
                all_projections["embeddings"][f"{proj_base}__pls12-{year_transform}"] = X_proj[:, [0, 1]].tolist()
                all_projections["embeddings"][f"{proj_base}__pls23-{year_transform}"] = X_proj[:, [1, 2]].tolist()
                all_projections["embeddings"][f"{proj_base}__pls34-{year_transform}"] = X_proj[:, [2, 3]].tolist()

        if "ruler" in targets:
            config_key = f"tfidf__{cleaning}__na__L00__ruler"
            print(f"  {config_key}...", flush=True)

            metrics_per_k = {}
            for k in K_VALUES:
                print(f"    k={k} (ruler PLS-DA)...", flush=True)
                metrics_per_k[str(k)] = fit_plsda_stratified_kfold(
                    X_labeled, y_ruler, n_components=k, n_splits=N_SPLITS)

            best_k = max(K_VALUES, key=lambda k: metrics_per_k[str(k)]["macro_f1_mean"])

            all_results[config_key] = {
                "method": "tfidf", "cleaning": cleaning, "pooling": "na",
                "layer": 0, "target": "ruler",
                "n_labeled": n_labeled,
                "metrics_per_k": metrics_per_k,
                "best_k_by_macro_f1": best_k,
            }

            model_da = fit_plsda_full(X_labeled, y_ruler, n_components=N_COMPONENTS_FULL)
            X_proj_da = project(model_da, X_labeled)   # project labeled only for TF-IDF
            all_projections["embeddings"][f"{proj_base}__plsda12"] = X_proj_da[:, [0, 1]].tolist()

            best_acc = metrics_per_k[str(best_k)]["accuracy_mean"]
            best_f1  = metrics_per_k[str(best_k)]["macro_f1_mean"]
            print(f"  ruler best_k={best_k} acc={best_acc:.3f} macro_f1={best_f1:.3f}")

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
