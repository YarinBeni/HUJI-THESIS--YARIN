#!/usr/bin/env python3
"""
02_merge_coords.py — Merge all embedding coords into seal_viz_data.json.

Sources:
  seal_viz_data.json          (base: SEAL fragments + TF-IDF keys)
  seal_round4/seal_qwen_coords.json       (Qwen + Random SEAL mean: 232 keys)
  seal_round4/seal_mlm_coords.json        (Akkadian MLM: 10 keys)
  seal_round4/seal_qwen_coords_last.json  (Qwen + Random SEAL last-token, if exists)
  orcc_round1/orcc_qwen_coords_mean.json  (ORCC mean-pooled, if exists)
  orcc_round1/orcc_qwen_coords_last.json  (ORCC last-token, if exists)

Also appends ORCC fragments from orcc_corpus.parquet if the file exists.

IMPORTANT: do NOT re-run 01_compute_tfidf_coords.py after this — it overwrites
seal_viz_data.json and erases the Qwen/MLM keys.

Usage (from repo root):
    python v_1/src/viz/02_merge_coords.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[3]
VIZ_DIR    = REPO_ROOT / "v_1/src/viz"
RESULTS    = REPO_ROOT / "v_1/src/linear_probing/results/seal_round4"
ORCC_RES   = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round1"

BASE_JSON  = VIZ_DIR  / "seal_viz_data.json"
QWEN_JSON  = RESULTS  / "seal_qwen_coords.json"
MLM_JSON   = RESULTS  / "seal_mlm_coords.json"
SEAL_LAST_JSON  = RESULTS  / "seal_qwen_coords_last.json"
ORCC_MEAN_JSON  = ORCC_RES / "orcc_qwen_coords_mean.json"
ORCC_LAST_JSON  = ORCC_RES / "orcc_qwen_coords_last.json"
ORCC_TFIDF_JSON = ORCC_RES / "orcc_tfidf_coords.json"
ORCC_MLM_JSON   = ORCC_RES / "orcc_mlm_coords.json"
ORCC_PARQUET    = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"


def load_coords(path: Path, label: str) -> dict:
    """Load a coords JSON; unwrap 'embeddings' wrapper if present."""
    data = json.loads(path.read_text())
    if "embeddings" in data and isinstance(data["embeddings"], dict):
        coords = data["embeddings"]
        print(f"  {label}: {len(coords)} keys (unwrapped from 'embeddings')")
    elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        coords = data
        print(f"  {label}: {len(coords)} keys (flat)")
    else:
        raise ValueError(f"Unexpected structure in {path.name}: top-level keys={list(data.keys())[:5]}")
    return coords


def pad_coords(coords, n_before, n_after):
    """Pad a coord list with [None, None] sentinels before and/or after.

    The HTML render() must skip null entries: if (!rawCoords[i]) continue;
    """
    return [[None, None]] * n_before + coords + [[None, None]] * n_after


def validate(embeddings: dict, n: int):
    errors = []
    for key, vals in embeddings.items():
        if len(vals) != n:
            errors.append(f"{key}: expected {n} rows, got {len(vals)}")
            continue
        non_null = [v for v in vals if v is not None and v[0] is not None]
        if non_null:
            flat = np.array(non_null, dtype=float)
            if flat.shape[1] != 2:
                errors.append(f"{key}: expected 2 columns, got {flat.shape[1]}")
            if np.isnan(flat).any():
                errors.append(f"{key}: NaN detected")
            if np.isinf(flat).any():
                errors.append(f"{key}: Inf detected")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError(f"{len(errors)} validation error(s) — see above")


def main():
    # Load base (TF-IDF keys + existing SEAL fragments)
    print("[1/6] Loading base (TF-IDF + SEAL fragments)...")
    base = json.loads(BASE_JSON.read_text())

    # Separate SEAL/DLL/LBPL fragments from any previously-added ORCC entries
    seal_fragments = [f for f in base["fragments"] if f.get("corpus") != "orcc"]
    n_seal = len(seal_fragments)
    print(f"  SEAL/DLL/LBPL fragments: {n_seal}")

    # Add year=null, ruler=null to SEAL fragments if not already present
    for frag in seal_fragments:
        if "year" not in frag:
            frag["year"] = None
        if "ruler" not in frag:
            frag["ruler"] = None

    tfidf_keys = list(base["embeddings"].keys())
    print(f"  Existing embedding keys: {len(tfidf_keys)}")

    # Load ORCC fragments if parquet exists
    print("\n[2/6] Loading ORCC fragments...")
    orcc_fragments = []
    if ORCC_PARQUET.exists():
        orcc_df = pd.read_parquet(ORCC_PARQUET)
        for _, row in orcc_df.iterrows():
            text_tier0 = str(row.get("text_tier0", "")) if pd.notna(row.get("text_tier0")) else ""
            snippet = " ".join(text_tier0.split()[:15])
            orcc_fragments.append({
                "fragment_id": str(row["fragment_id"]),
                "corpus": "orcc",
                "word_language": str(row["word_language"]) if pd.notna(row.get("word_language")) else None,
                "domain": "ORCC",
                "ruler": str(row["ruler"]) if pd.notna(row.get("ruler")) else None,
                "period": str(row["period"]) if pd.notna(row.get("period")) else None,
                "genre": str(row["genre"]) if pd.notna(row.get("genre")) else None,
                "sub_genre": str(row["sub_genre"]) if pd.notna(row.get("sub_genre")) else None,
                "provenance": str(row["provenance"]) if pd.notna(row.get("provenance")) else None,
                "sub_provenance": str(row["sub_provenance"]) if pd.notna(row.get("sub_provenance")) else None,
                "word_count": int(row["word_count"]) if pd.notna(row.get("word_count")) else 0,
                "text_snippet": snippet,
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            })
        print(f"  ORCC fragments loaded: {len(orcc_fragments)}")
    else:
        print(f"  (orcc_corpus.parquet not found — no ORCC fragments)")

    n_orcc = len(orcc_fragments)
    n_total = n_seal + n_orcc
    all_fragments = seal_fragments + orcc_fragments
    base["fragments"] = all_fragments
    print(f"  Total fragments: {n_total} (SEAL={n_seal}, ORCC={n_orcc})")

    # Load Qwen + Random (SEAL mean)
    print("\n[3/6] Loading Qwen + Random coords (SEAL mean)...")
    qwen_coords = load_coords(QWEN_JSON, "seal_qwen_coords.json")

    # Load MLM
    print("\n[4/6] Loading MLM coords...")
    mlm_coords = load_coords(MLM_JSON, "seal_mlm_coords.json")

    # Merge existing SEAL keys and pad for ORCC fragments
    print("\n[5/6] Merging SEAL keys and padding for ORCC...")
    merged = {}
    for key, vals in base["embeddings"].items():
        merged[key] = pad_coords(vals, 0, n_orcc)
    for key, vals in qwen_coords.items():
        merged[key] = pad_coords(vals, 0, n_orcc)
    for key, vals in mlm_coords.items():
        merged[key] = pad_coords(vals, 0, n_orcc)
    n_seal_keys = len(merged)
    print(f"  SEAL keys after padding: {n_seal_keys}")

    # Load SEAL last-token keys (if exists), pad for ORCC
    n_seal_last_keys = 0
    if SEAL_LAST_JSON.exists():
        print(f"  Loading {SEAL_LAST_JSON.name}...")
        seal_last_coords = load_coords(SEAL_LAST_JSON, SEAL_LAST_JSON.name)
        for key, vals in seal_last_coords.items():
            merged[key] = pad_coords(vals, 0, n_orcc)
        n_seal_last_keys = len(seal_last_coords)
    else:
        print(f"  (seal_qwen_coords_last.json not found — skipping)")

    # Load ORCC mean keys (if exists): merge into SAME key as SEAL.
    # Each SEAL key was padded with [None,None]*n_orcc — replace that padding
    # with real ORCC coords so the shared array covers all n_total fragments.
    n_orcc_mean_keys = 0
    if ORCC_MEAN_JSON.exists():
        print(f"  Loading {ORCC_MEAN_JSON.name}...")
        orcc_mean_coords = load_coords(ORCC_MEAN_JSON, ORCC_MEAN_JSON.name)
        for key, vals in orcc_mean_coords.items():
            if key in merged:
                merged[key] = merged[key][:n_seal] + vals
            else:
                merged[key] = pad_coords(vals, n_seal, 0)
        n_orcc_mean_keys = len(orcc_mean_coords)
    else:
        print(f"  (orcc_qwen_coords_mean.json not found — skipping)")

    # Load ORCC last-token keys: same logic — merge into the shared key.
    n_orcc_last_keys = 0
    if ORCC_LAST_JSON.exists():
        print(f"  Loading {ORCC_LAST_JSON.name}...")
        orcc_last_coords = load_coords(ORCC_LAST_JSON, ORCC_LAST_JSON.name)
        for key, vals in orcc_last_coords.items():
            if key in merged:
                merged[key] = merged[key][:n_seal] + vals
            else:
                merged[key] = pad_coords(vals, n_seal, 0)
        n_orcc_last_keys = len(orcc_last_coords)
    else:
        print(f"  (orcc_qwen_coords_last.json not found — skipping)")

    # Load ORCC TF-IDF coords: replace the null-padding that was added for SEAL tfidf keys.
    n_orcc_tfidf_keys = 0
    if ORCC_TFIDF_JSON.exists():
        print(f"  Loading {ORCC_TFIDF_JSON.name}...")
        orcc_tfidf_coords = load_coords(ORCC_TFIDF_JSON, ORCC_TFIDF_JSON.name)
        for key, vals in orcc_tfidf_coords.items():
            if key in merged:
                merged[key] = merged[key][:n_seal] + vals
            else:
                merged[key] = pad_coords(vals, n_seal, 0)
        n_orcc_tfidf_keys = len(orcc_tfidf_coords)
    else:
        print(f"  (orcc_tfidf_coords.json not found — skipping)")

    # Load ORCC MLM coords: merge into shared mlm keys (or create ORCC-only with SEAL padding).
    n_orcc_mlm_keys = 0
    if ORCC_MLM_JSON.exists():
        print(f"  Loading {ORCC_MLM_JSON.name}...")
        orcc_mlm_coords = load_coords(ORCC_MLM_JSON, ORCC_MLM_JSON.name)
        for key, vals in orcc_mlm_coords.items():
            if key in merged:
                merged[key] = merged[key][:n_seal] + vals
            else:
                merged[key] = pad_coords(vals, n_seal, 0)
        n_orcc_mlm_keys = len(orcc_mlm_coords)
    else:
        print(f"  (orcc_mlm_coords.json not found — skipping)")

    # Validate all keys have n_total rows
    print("\n[6/6] Validating and saving...")
    validate(merged, n_total)

    base["embeddings"] = merged

    # Summary
    methods = sorted({k.split("__")[0] for k in merged})
    print(f"\n  Total fragments: {n_total}")
    print(f"  Total embedding keys: {len(merged)}")
    print(f"  Key breakdown:")
    print(f"    existing SEAL       : {n_seal_keys}")
    print(f"    SEAL-last           : {n_seal_last_keys}")
    print(f"    ORCC-mean           : {n_orcc_mean_keys}")
    print(f"    ORCC-last           : {n_orcc_last_keys}")
    print(f"    ORCC-tfidf          : {n_orcc_tfidf_keys}")
    print(f"    ORCC-mlm            : {n_orcc_mlm_keys}")
    print(f"  Methods: {methods}")

    BASE_JSON.write_text(json.dumps(base))
    size_kb = BASE_JSON.stat().st_size / 1024
    print(f"\nSaved {BASE_JSON.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    print("Done")


if __name__ == "__main__":
    main()
