"""
01b_compute_tfidf_orcc_coords.py — TF-IDF 2D coordinates for ORCC fragments.

Fits the same TF-IDF vectorizer as 01_compute_tfidf_coords.py (char_wb 2–5
n-grams, trained on SEAL texts only) then transforms ORCC texts and runs
t-SNE + PCA to produce 2D coords for 1202 ORCC fragments.

Output: v_1/src/linear_probing/results/orcc_round1/orcc_tfidf_coords.json
  {"tfidf__tier0__na__tsne":   [[x,y], ...],   ← 1202 entries
   "tfidf__tier0__na__pca":    [[x,y], ...],
   "tfidf__maximal__na__tsne": [[x,y], ...],
   "tfidf__maximal__na__pca":  [[x,y], ...]}

Run from repo root:
  python3 v_1/src/viz/01b_compute_tfidf_orcc_coords.py
"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT    = pathlib.Path(__file__).resolve().parents[3]
SEAL_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
ORCC_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_JSON     = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round1/orcc_tfidf_coords.json"

TSNE_PARAMS  = dict(n_components=2, perplexity=30, max_iter=1000, random_state=42)
PCA_PARAMS   = dict(n_components=2, random_state=42)
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))


def compute_coords(matrix, label: str) -> dict:
    arr = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    print(f"  t-SNE on {label} ({arr.shape})…", flush=True)
    tsne_coords = TSNE(**TSNE_PARAMS).fit_transform(arr)
    print(f"  PCA  on {label}…", flush=True)
    pca_coords  = PCA(**PCA_PARAMS).fit_transform(arr)
    return {"tsne": tsne_coords.tolist(), "pca": pca_coords.tolist()}


def main():
    print(f"Loading SEAL parquet (to fit TF-IDF vectorizer)…")
    seal_df = pd.read_parquet(SEAL_PARQUET)
    print(f"  {len(seal_df)} SEAL fragments")

    print(f"Loading ORCC parquet…")
    orcc_df = pd.read_parquet(ORCC_PARQUET)
    n_orcc = len(orcc_df)
    print(f"  {n_orcc} ORCC fragments")

    embeddings = {}

    # tier0: fit on SEAL, transform ORCC
    print("\nFitting TF-IDF on SEAL text_tier0…")
    vec_tier0 = TfidfVectorizer(**TFIDF_PARAMS)
    vec_tier0.fit(seal_df["text_tier0"].astype(str))
    print(f"  vocab size: {len(vec_tier0.vocabulary_)}")
    orcc_tier0_mat = vec_tier0.transform(orcc_df["text_tier0"].astype(str))
    print(f"  ORCC transform shape: {orcc_tier0_mat.shape}")
    coords_tier0 = compute_coords(orcc_tier0_mat, "ORCC tier0")
    embeddings["tfidf__tier0__na__tsne"] = coords_tier0["tsne"]
    embeddings["tfidf__tier0__na__pca"]  = coords_tier0["pca"]

    # maximal: fit on SEAL, transform ORCC
    print("\nFitting TF-IDF on SEAL text_maximal…")
    vec_maximal = TfidfVectorizer(**TFIDF_PARAMS)
    vec_maximal.fit(seal_df["text_maximal"].astype(str))
    print(f"  vocab size: {len(vec_maximal.vocabulary_)}")
    orcc_maximal_mat = vec_maximal.transform(orcc_df["text_maximal"].astype(str))
    print(f"  ORCC transform shape: {orcc_maximal_mat.shape}")
    coords_maximal = compute_coords(orcc_maximal_mat, "ORCC maximal")
    embeddings["tfidf__maximal__na__tsne"] = coords_maximal["tsne"]
    embeddings["tfidf__maximal__na__pca"]  = coords_maximal["pca"]

    # Validate
    print("\n=== Validation ===")
    for key, vals in embeddings.items():
        assert len(vals) == n_orcc, f"{key}: expected {n_orcc}, got {len(vals)}"
        flat = np.array(vals, dtype=float)
        assert flat.shape[1] == 2, f"{key}: expected 2 columns"
        assert not np.isnan(flat).any(), f"{key}: NaN detected"
        print(f"  {key}: {len(vals)} pairs ✓")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    size_kb = OUT_JSON.stat().st_size / 1024
    print(f"\nSaved {OUT_JSON} ({size_kb:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
