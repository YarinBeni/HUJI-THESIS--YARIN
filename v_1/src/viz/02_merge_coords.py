#!/usr/bin/env python3
"""
02_merge_coords.py — Plan E: merge all embedding coords into seal_viz_data.json.

Sources:
  seal_viz_data.json          (base: fragments + 4 TF-IDF keys)
  seal_round4/seal_qwen_coords.json   (Qwen + Random Qwen: 232 keys, nested under "embeddings")
  seal_round4/seal_mlm_coords.json    (Akkadian MLM: 10 keys, flat)

Output:
  seal_viz_data.json  (merged: fragments + all embedding keys)

IMPORTANT: do NOT re-run 01_compute_tfidf_coords.py after this — it overwrites
seal_viz_data.json and erases the Qwen/MLM keys.

Usage (from repo root):
    python v_1/src/viz/02_merge_coords.py
"""

import json
import numpy as np
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[3]
VIZ_DIR    = REPO_ROOT / "v_1/src/viz"
RESULTS    = REPO_ROOT / "v_1/src/linear_probing/results/seal_round4"

BASE_JSON  = VIZ_DIR    / "seal_viz_data.json"
QWEN_JSON  = RESULTS    / "seal_qwen_coords.json"
MLM_JSON   = RESULTS    / "seal_mlm_coords.json"


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


def validate(embeddings: dict, n: int):
    errors = []
    for key, vals in embeddings.items():
        if len(vals) != n:
            errors.append(f"{key}: expected {n} rows, got {len(vals)}")
            continue
        flat = np.array(vals, dtype=float)
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
    n_frags = 384

    # Load base
    print("[1/4] Loading base (TF-IDF)...")
    base = json.loads(BASE_JSON.read_text())
    assert len(base["fragments"]) == n_frags, \
        f"Expected {n_frags} fragments, got {len(base['fragments'])}"
    tfidf_keys = list(base["embeddings"].keys())
    print(f"  fragments: {len(base['fragments'])}, existing keys: {tfidf_keys}")

    # Load Qwen + Random
    print("\n[2/4] Loading Qwen + Random coords...")
    qwen_coords = load_coords(QWEN_JSON, "seal_qwen_coords.json")

    # Load MLM
    print("\n[3/4] Loading MLM coords...")
    mlm_coords = load_coords(MLM_JSON, "seal_mlm_coords.json")

    # Merge
    print("\n[4/4] Merging and validating...")
    merged = dict(base["embeddings"])   # start with TF-IDF keys
    merged.update(qwen_coords)
    merged.update(mlm_coords)

    validate(merged, n_frags)

    base["embeddings"] = merged

    # Summary
    methods = sorted({k.split("__")[0] for k in merged})
    print(f"\n  Total keys : {len(merged)}")
    print(f"  Methods    : {methods}")
    for method in methods:
        method_keys = [k for k in merged if k.startswith(method + "__")]
        print(f"    {method:<8}: {len(method_keys)} keys")

    BASE_JSON.write_text(json.dumps(base))
    size_kb = BASE_JSON.stat().st_size / 1024
    print(f"\nSaved {BASE_JSON.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    print("✅ Done")


if __name__ == "__main__":
    main()
