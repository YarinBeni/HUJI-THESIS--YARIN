"""Shared plumbing for chrono/ — the only file every module may import.

Holds the three things the INTERFACES.md contract centralizes so that six
parallel builders cannot drift: the time convention (astronomical years,
larger = later), canonical paths, and the results writer. Anything larger
belongs to the owning module.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHRONO = os.path.join(REPO, "chrono")
ART = os.path.join(CHRONO, "artifacts")
ORCC = os.path.join(REPO, "v_1", "data", "evaluation", "corpora",
                    "orcc_corpus.parquet")
PAIRS_DIR = os.path.join(REPO, "v_1", "src", "phase2", "pairs")

RESULTS_COLS = ["run_id", "git_sha", "config_sha", "seed", "split",
                "metric", "value", "n", "extra"]


def to_astro(year_bc):
    """BC-positive (631 = 631 BC, larger = earlier) -> astronomical t
    (larger = LATER). The single sanctioned conversion point (SLA section 1).
    """
    return -np.asarray(year_bc, dtype=float)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_sha(cfg: dict) -> str:
    return sha16(json.dumps(cfg, sort_keys=True, default=str))


def append_results(rows: list) -> str:
    """Append rows (dicts with RESULTS_COLS keys; extra is a JSON string)
    to artifacts/results.parquet. Returns the path."""
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "results.parquet")
    df = pd.DataFrame(rows)
    missing = set(RESULTS_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"results rows missing {sorted(missing)}")
    df = df[RESULTS_COLS]
    if os.path.exists(p):
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
    df.to_parquet(p, index=False)
    return p
