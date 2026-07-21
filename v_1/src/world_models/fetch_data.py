"""Re-download the six Gurnee & Tegmark entity CSVs and verify the vendored copies.

The CSVs under data/entity_datasets/ are committed (23 MB) so the cluster jobs never
depend on GitHub being reachable. This script exists to (a) re-fetch them if the
upstream repo moves, (b) verify the vendored copies still match the expected row
counts from the paper. Vendored from wesg52/world-models @ a572f162948e (2026-07-21).

Usage:
    python fetch_data.py            # verify vendored copies
    python fetch_data.py --fetch    # re-download from GitHub, then verify
"""
import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "entity_datasets")
RAW_BASE = ("https://raw.githubusercontent.com/wesg52/world-models/"
            "a572f162948ee185e9c842eb5fa15e23aad3d218/data/entity_datasets")

# entity_type -> expected raw CSV lines minus header, from the upstream repo.
# NB headline contains quoted newlines: 28461 raw lines parse to 28,389 rows
# (the paper's count); this check is file-integrity, not parsed-row count.
EXPECTED_ROWS = {
    "world_place": 39585,
    "us_place": 29997,
    "nyc_place": 19838,
    "historical_figure": 37539,
    "art": 31321,
    "headline": 28461,
}


def fetch(entity_type: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{RAW_BASE}/{entity_type}.csv"
    dest = os.path.join(DATA_DIR, f"{entity_type}.csv")
    print(f"[fetch] {url}")
    urllib.request.urlretrieve(url, dest)


def verify(entity_type: str) -> bool:
    path = os.path.join(DATA_DIR, f"{entity_type}.csv")
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return False
    with open(path, "rb") as f:
        n = sum(1 for _ in f) - 1
    ok = n == EXPECTED_ROWS[entity_type]
    print(f"[{'ok' if ok else 'BAD'}] {entity_type}: {n} rows "
          f"(expected {EXPECTED_ROWS[entity_type]})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download before verifying")
    args = ap.parse_args()
    if args.fetch:
        for et in EXPECTED_ROWS:
            fetch(et)
    sys.exit(0 if all(verify(et) for et in EXPECTED_ROWS) else 1)
