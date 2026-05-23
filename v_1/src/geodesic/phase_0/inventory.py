#!/usr/bin/env python3
"""Phase 0: activation inventory check.

Run directly on the cluster login node (no sbatch needed — pure filesystem + tiny NPZ reads).
Output: v_1/src/geodesic/results/phase_0_inventory.json

Usage (from repo root on cluster):
    python v_1/src/geodesic/phase_0/inventory.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root
ACTS_BASE = ROOT / "v_1/src/linear_probing/results/orcc__embed/activations"
ACTS_FALLBACK = ROOT / "v_1/src/linear_probing/results/orcc_round1/activations"
ACTS_FALLBACK2 = ROOT / "v_1/src/linear_probing/results/activations"
PARQUET_PATH = ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT_JSON = ROOT / "v_1/src/geodesic/results/phase_0_inventory.json"

# Expected combos per non-random method
NON_RANDOM_METHODS = [
    ("qwen",                  ["tier0", "maximal"], ["mean", "last"]),
    ("mlm",                   ["tier0"],            ["mean"]),        # Aeneas: tier0/mean only
    ("thalesian_akk300m",     ["tier0", "maximal"], ["mean", "last"]),
    ("thalesian_cunei400m",   ["tier0", "maximal"], ["mean", "last"]),
]

# Random-Qwen: same combos theoretically but path may differ
RANDOM_COMBOS = [
    ("tier0",   "mean"),
    ("tier0",   "last"),
    ("maximal", "mean"),
    ("maximal", "last"),
]

def find_npz_files(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("layer_*.npz"))


def spot_check_npz(directory: Path, npz_name: str) -> dict:
    """Load one NPZ and return shape + dtype."""
    try:
        import numpy as np
        arr = np.load(directory / npz_name)
        keys = list(arr.files)
        if not keys:
            return {"error": "empty npz"}
        data = arr[keys[0]]
        return {"shape": list(data.shape), "dtype": str(data.dtype), "key": keys[0]}
    except Exception as e:
        return {"error": str(e)}


def check_parquet() -> dict:
    try:
        import pandas as pd
        df = pd.read_parquet(PARQUET_PATH)
        year_labeled = int(df["year"].notna().sum())
        return {
            "path": str(PARQUET_PATH),
            "exists": True,
            "total_rows": len(df),
            "year_labeled": year_labeled,
            "columns": list(df.columns),
        }
    except Exception as e:
        return {"path": str(PARQUET_PATH), "exists": False, "error": str(e)}


def check_method(method: str, cleaning: str, pool: str) -> dict:
    """Check primary + fallback paths for one (method, cleaning, pool) combo."""
    dir_name = f"{method}_{cleaning}_{pool}"
    candidates = [
        ACTS_BASE / dir_name,
        ACTS_FALLBACK / dir_name,
    ]
    # Extra fallback for qwen-based methods
    if method in ("qwen", "random"):
        hf_variant = "qwen2.5-7b-instruct-random" if method == "random" else "qwen2.5-7b-instruct"
        candidates.append(ACTS_FALLBACK2 / hf_variant / cleaning)

    found_path = None
    for cand in candidates:
        layers = find_npz_files(cand)
        if layers:
            found_path = cand
            found_layers = layers
            break
    else:
        found_layers = []

    rec = {
        "method": method,
        "cleaning": cleaning,
        "pool": pool,
        "found": found_path is not None,
        "path": str(found_path) if found_path else None,
        "n_layers": len(found_layers),
        "layers": found_layers[:3] + (["..."] if len(found_layers) > 3 else []),
        "candidates_checked": [str(c) for c in candidates],
    }
    if found_path and found_layers:
        # spot-check first layer
        rec["spot_check"] = spot_check_npz(found_path, found_layers[0])
    return rec


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    inventory = {
        "parquet": check_parquet(),
        "activations": [],
        "random_qwen": [],
        "summary": {},
    }

    # Non-random methods
    for method, cleanings, pools in NON_RANDOM_METHODS:
        for cleaning in cleanings:
            for pool in pools:
                rec = check_method(method, cleaning, pool)
                inventory["activations"].append(rec)

    # Random-Qwen with all three candidate paths explicitly listed
    for cleaning, pool in RANDOM_COMBOS:
        rec = check_method("random", cleaning, pool)
        inventory["random_qwen"].append(rec)

    # Summary stats
    all_recs = inventory["activations"] + inventory["random_qwen"]
    n_total = len(all_recs)
    n_found = sum(1 for r in all_recs if r["found"])
    missing = [f"{r['method']}_{r['cleaning']}_{r['pool']}" for r in all_recs if not r["found"]]

    inventory["summary"] = {
        "n_combos_total": n_total,
        "n_found": n_found,
        "n_missing": n_total - n_found,
        "missing_combos": missing,
        "parquet_ok": inventory["parquet"].get("year_labeled", 0) >= 1190,
        "thalesian_cunei400m_tier0_mean_ok": any(
            r["found"]
            for r in inventory["activations"]
            if r["method"] == "thalesian_cunei400m"
            and r["cleaning"] == "tier0"
            and r["pool"] == "mean"
        ),
        "random_qwen_any_found": any(r["found"] for r in inventory["random_qwen"]),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(inventory, f, indent=2)

    print(f"Inventory written to {OUT_JSON}")
    print(f"\nSummary:")
    print(f"  Parquet: {inventory['parquet'].get('year_labeled', 'ERR')} year-labeled rows")
    print(f"  Activation combos found: {n_found}/{n_total}")
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    thal_ok = inventory["summary"]["thalesian_cunei400m_tier0_mean_ok"]
    rqwen_ok = inventory["summary"]["random_qwen_any_found"]
    print(f"\n  Phase A gate (Thalesian cunei400m tier0/mean found): {'PASS' if thal_ok else 'FAIL'}")
    print(f"  Random-Qwen any path found: {'YES' if rqwen_ok else 'NO — needs re-extraction or drop'}")
    print(f"\n  Phase 0: {'PASS' if thal_ok else 'BLOCKED — cannot proceed to Phase A'}")


if __name__ == "__main__":
    main()
