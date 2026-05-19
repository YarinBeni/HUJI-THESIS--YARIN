"""build_balanced_subset.py — Round 2 Phase 0: Monte Carlo balanced draws from ORCC corpus.

Restricts to 8 rulers with >15 fragments and performs MC undersampling:
  - N draws, each sampling k=21 fragments per ruler without replacement.
  - Enables averaging Macro-F1 across draws to control for class imbalance.

Usage (all defaults):
    python build_balanced_subset.py

Usage (custom):
    python build_balanced_subset.py --corpus path/to/orcc_corpus.parquet \
        --n_draws 200 --k 21 --seed 42 --out_dir path/to/output/dir
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact ORCC corpus spellings for the 8 rulers with >15 fragments
RULERS_8 = [
    "Ashurbanipal",        # 268 frags
    "Sennacherib",         # 237 frags
    "Esarhaddon",          # 176 frags
    "Sargon II",           # 144 frags
    "Nebuchadnezzar II",   # 87 frags
    "Tiglath-pileser III", # 75 frags
    "Nabonidus",           # 68 frags
    "Sîn-šarru-iškun",    # 21 frags
]

_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]  # lititure-review/

DEFAULT_CORPUS = _REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_OUT_DIR = _REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
DEFAULT_N_DRAWS = 200
DEFAULT_K = 21
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_balanced_draws(
    corpus_path: pathlib.Path,
    n_draws: int,
    k: int,
    seed_base: int,
    out_dir: pathlib.Path,
) -> None:
    """Build MC balanced draws and write all output files."""

    # ------------------------------------------------------------------
    # 1. Load and filter corpus
    # ------------------------------------------------------------------
    df = pd.read_parquet(corpus_path)

    # Build per-ruler index maps over the 8 rulers
    ruler_to_ids: dict[str, list[str]] = {}
    for ruler in RULERS_8:
        mask = df["ruler"] == ruler
        ids = df.loc[mask, "fragment_id"].tolist()
        ruler_to_ids[ruler] = ids

    # Assertions: each ruler must have >= k fragments
    for ruler in RULERS_8:
        n = len(ruler_to_ids[ruler])
        assert n >= k, (
            f"Ruler '{ruler}' has only {n} fragments, need >= {k}."
        )

    n_frags_total = len(df)
    all_fragment_ids: list[str] = df["fragment_id"].tolist()

    # Map fragment_id -> column index in the full corpus (row order as loaded)
    frag_id_to_col: dict[str, int] = {fid: i for i, fid in enumerate(all_fragment_ids)}

    # ------------------------------------------------------------------
    # 2. Create output directories
    # ------------------------------------------------------------------
    draws_dir = out_dir / "draws"
    draws_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Build draws matrix and per-draw JSON files
    # ------------------------------------------------------------------
    draws_matrix = np.zeros((n_draws, n_frags_total), dtype=bool)

    all_draws_ids: list[list[str]] = []

    for i in range(n_draws):
        rng = np.random.default_rng(seed_base + i)
        draw_ids: list[str] = []
        ruler_counts: dict[str, int] = {}

        for ruler in RULERS_8:
            ids = ruler_to_ids[ruler]
            sampled = rng.choice(ids, size=k, replace=False).tolist()
            draw_ids.extend(sampled)
            ruler_counts[ruler] = len(sampled)

        all_draws_ids.append(draw_ids)

        # Fill draws_matrix row
        for fid in draw_ids:
            col = frag_id_to_col[fid]
            draws_matrix[i, col] = True

        # Write per-draw JSON
        draw_record = {
            "draw_id": i,
            "seed": seed_base + i,
            "fragment_ids": draw_ids,
            "ruler_counts": ruler_counts,
        }
        draw_path = draws_dir / f"draw_{i:04d}.json"
        with open(draw_path, "w", encoding="utf-8") as f:
            json.dump(draw_record, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 4. Write manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "n_draws": n_draws,
        "k": k,
        "n_rulers": len(RULERS_8),
        "rulers": RULERS_8,
        "total_frags_per_draw": k * len(RULERS_8),
        "corpus_path": str(corpus_path),
        "seed_base": seed_base,
        "produced_by": "build_balanced_subset.py",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 5. Write draws_matrix.npy
    # ------------------------------------------------------------------
    np.save(out_dir / "draws_matrix.npy", draws_matrix)

    # ------------------------------------------------------------------
    # 6. Write corpus_fragment_order.json sidecar
    # ------------------------------------------------------------------
    with open(out_dir / "corpus_fragment_order.json", "w", encoding="utf-8") as f:
        json.dump(all_fragment_ids, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    total_frags_per_draw = k * len(RULERS_8)
    print("=" * 60)
    print("build_balanced_subset.py — Summary")
    print("=" * 60)
    print(f"  Corpus path      : {corpus_path}")
    print(f"  n_draws          : {n_draws}")
    print(f"  k (per ruler)    : {k}")
    print(f"  n_rulers         : {len(RULERS_8)}")
    print(f"  total per draw   : {total_frags_per_draw}")
    print(f"  draws_matrix     : shape={draws_matrix.shape}, dtype={draws_matrix.dtype}")
    print(f"  Output dir       : {out_dir}")
    print()

    # ------------------------------------------------------------------
    # 8. Sanity checks
    # ------------------------------------------------------------------
    print("--- Sanity checks ---")

    # Draw 0: first 5 fragment_ids + ruler breakdown
    draw0 = all_draws_ids[0]
    print(f"Draw 0 — first 5 fragment_ids: {draw0[:5]}")
    # Recompute ruler counts for draw 0
    d0_ruler_counts: dict[str, int] = {r: 0 for r in RULERS_8}
    for fid in draw0:
        for ruler in RULERS_8:
            if fid in ruler_to_ids[ruler]:
                d0_ruler_counts[ruler] += 1
                break
    print(f"Draw 0 — ruler counts: {d0_ruler_counts}")
    all_21 = all(v == k for v in d0_ruler_counts.values())
    print(f"Draw 0 — all rulers have exactly {k} frags: {all_21}")
    print()

    # Draw 0 vs Draw 1 overlap (per-ruler)
    draw1 = all_draws_ids[1]
    set0 = set(draw0)
    set1 = set(draw1)
    overlap_total = len(set0 & set1)
    print(f"Draw 0 vs Draw 1 — total overlap: {overlap_total} / {total_frags_per_draw} "
          f"({overlap_total / total_frags_per_draw:.1%})")

    # Per-ruler overlap
    # Build ruler -> set of ids for draw 0 and draw 1
    ruler_set0: dict[str, set] = {r: set() for r in RULERS_8}
    ruler_set1: dict[str, set] = {r: set() for r in RULERS_8}
    for fid in draw0:
        for ruler in RULERS_8:
            if fid in ruler_to_ids[ruler]:
                ruler_set0[ruler].add(fid)
                break
    for fid in draw1:
        for ruler in RULERS_8:
            if fid in ruler_to_ids[ruler]:
                ruler_set1[ruler].add(fid)
                break
    print("Draw 0 vs Draw 1 — per-ruler overlap:")
    for ruler in RULERS_8:
        ov = len(ruler_set0[ruler] & ruler_set1[ruler])
        print(f"  {ruler:25s}: {ov}/{k}")
    print()

    # draws_matrix checks
    print(f"draws_matrix shape : {draws_matrix.shape}  (expected: ({n_draws}, {n_frags_total}))")
    print(f"draws_matrix dtype : {draws_matrix.dtype}")
    row_sums = draws_matrix.sum(axis=1)
    print(f"Row sums — min={row_sums.min()}, max={row_sums.max()}, all=={total_frags_per_draw}: "
          f"{bool((row_sums == total_frags_per_draw).all())}")
    print("=" * 60)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build balanced MC draws from ORCC corpus (Round 2 Phase 0)."
    )
    p.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=DEFAULT_CORPUS,
        help=f"Path to orcc_corpus.parquet (default: {DEFAULT_CORPUS})",
    )
    p.add_argument(
        "--n_draws",
        type=int,
        default=DEFAULT_N_DRAWS,
        help=f"Number of MC draws (default: {DEFAULT_N_DRAWS})",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Fragments per ruler per draw (default: {DEFAULT_K})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed; draw i uses seed+i (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_balanced_draws(
        corpus_path=args.corpus.resolve(),
        n_draws=args.n_draws,
        k=args.k,
        seed_base=args.seed,
        out_dir=args.out_dir.resolve(),
    )
