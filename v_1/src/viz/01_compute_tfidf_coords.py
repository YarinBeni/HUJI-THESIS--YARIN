"""
Plan B — Step 1: Compute TF-IDF 2D coordinates for SEAL fragments.

Reads seal_corpus.parquet, fits TF-IDF char_wb(2,5) on text_tier0 and
text_maximal, runs t-SNE and PCA on each, and writes seal_viz_data.json
with 4 embedding keys:
  tfidf__tier0__na__tsne
  tfidf__tier0__na__pca
  tfidf__maximal__na__tsne
  tfidf__maximal__na__pca

Run from repo root:
  python3 v_1/src/viz/01_compute_tfidf_coords.py
"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
OUT_JSON = pathlib.Path(__file__).parent / "seal_viz_data.json"

TSNE_PARAMS = dict(n_components=2, perplexity=30, max_iter=1000, random_state=42)
PCA_PARAMS  = dict(n_components=2, random_state=42)
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))


def first_n_words(text: str, n: int = 15) -> str:
    return " ".join(str(text).split()[:n])


def compute_coords(matrix, label: str) -> dict:
    """Return {'tsne': [[x,y],...], 'pca': [[x,y],...]}."""
    print(f"  t-SNE on {label} ({matrix.shape})…", flush=True)
    tsne_coords = TSNE(**TSNE_PARAMS).fit_transform(matrix.toarray()
                       if hasattr(matrix, "toarray") else matrix)
    print(f"  PCA  on {label}…", flush=True)
    pca_coords  = PCA(**PCA_PARAMS).fit_transform(matrix.toarray()
                      if hasattr(matrix, "toarray") else matrix)
    return {
        "tsne": tsne_coords.tolist(),
        "pca":  pca_coords.tolist(),
    }


def main():
    print(f"Loading parquet from {PARQUET}…")
    df = pd.read_parquet(PARQUET)
    assert len(df) == 384, f"Expected 384 fragments, got {len(df)}"
    print(f"  {len(df)} fragments loaded.")

    # Build fragments metadata list
    fragments = []
    for _, row in df.iterrows():
        fragments.append({
            "fragment_id":  int(row["fragment_id"]),
            "corpus":       str(row["corpus"]),
            "domain":       str(row["domain"]),
            "period":       str(row["period"])    if pd.notna(row["period"])    else None,
            "genre":        str(row["genre"])     if pd.notna(row["genre"])     else None,
            "sub_genre":    str(row["sub_genre"]) if pd.notna(row["sub_genre"]) else None,
            "provenance":   str(row["provenance"]) if pd.notna(row["provenance"]) else None,
            "word_count":   int(row["word_count"]),
            "text_snippet": first_n_words(row["text_tier0"], 15),
        })

    embeddings = {}

    # TF-IDF on tier0
    print("Fitting TF-IDF on text_tier0…")
    vec_tier0 = TfidfVectorizer(**TFIDF_PARAMS)
    mat_tier0 = vec_tier0.fit_transform(df["text_tier0"].astype(str))
    print(f"  vocab size: {mat_tier0.shape[1]}")
    coords_tier0 = compute_coords(mat_tier0, "tier0")
    embeddings["tfidf__tier0__na__tsne"] = coords_tier0["tsne"]
    embeddings["tfidf__tier0__na__pca"]  = coords_tier0["pca"]

    # TF-IDF on maximal
    print("Fitting TF-IDF on text_maximal…")
    vec_maximal = TfidfVectorizer(**TFIDF_PARAMS)
    mat_maximal = vec_maximal.fit_transform(df["text_maximal"].astype(str))
    print(f"  vocab size: {mat_maximal.shape[1]}")
    coords_maximal = compute_coords(mat_maximal, "maximal")
    embeddings["tfidf__maximal__na__tsne"] = coords_maximal["tsne"]
    embeddings["tfidf__maximal__na__pca"]  = coords_maximal["pca"]

    output = {"fragments": fragments, "embeddings": embeddings}

    print(f"Writing {OUT_JSON}…")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    # Validate
    n_frags = len(output["fragments"])
    n_keys  = len(output["embeddings"])
    print(f"\n=== Validation ===")
    print(f"fragments: {n_frags}  (expected 384)")
    print(f"embedding keys: {n_keys}  (expected 4)")
    for key, coords in output["embeddings"].items():
        assert len(coords) == 384, f"{key}: expected 384 pairs, got {len(coords)}"
        print(f"  {key}: {len(coords)} pairs ✓")
    print("Done.")


if __name__ == "__main__":
    main()
