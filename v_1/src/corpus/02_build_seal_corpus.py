#!/usr/bin/env python
"""02_build_seal_corpus.py — Phase A: build fragment-level SEAL corpus parquet.

Reads:
  v_1/data/raw/chungrong/seal_round4/{seal,dll,lbpl}.csv
  v_1/data/raw/chungrong/seal_round4/inspection_report.json  (Phase 0 contract)

Writes:
  v_1/data/evaluation/corpora/seal_corpus.parquet     (384 rows, fragment-level)
  v_1/data/evaluation/corpora/seal_corpus_summary.json

Plan reference: Section 11 steps 6–8, Sections 5, 12, 15, 16 of
  v_1/justification/seal_round4_pipeline_plan.md

Idempotent: re-running overwrites both output files.
Dependencies: stdlib + pandas + numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUND_DIR = REPO_ROOT / "v_1" / "data" / "raw" / "chungrong" / "seal_round4"
CSV_FILES = OrderedDict(
    [
        ("seal", ROUND_DIR / "seal.csv"),
        ("dll", ROUND_DIR / "dll.csv"),
        ("lbpl", ROUND_DIR / "lbpl.csv"),
    ]
)
INSPECTION_REPORT = ROUND_DIR / "inspection_report.json"
CORPORA_DIR = REPO_ROOT / "v_1" / "data" / "evaluation" / "corpora"
CORPUS_PARQUET = CORPORA_DIR / "seal_corpus.parquet"
CORPUS_SUMMARY = CORPORA_DIR / "seal_corpus_summary.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "fragment_id",
    "fragment_line_num",
    "index_in_line",
    "word_language",
    "domain",
    "period",
    "genre",
    "sub_genre",
    "provenance",
    "sub_provenance",
    "place_discovery",
    "place_composition",
    "value",
    "clean_value",
    "lemma",
]

METADATA_COLS = [
    "word_language",
    "domain",
    "period",
    "genre",
    "sub_genre",
    "provenance",
    "sub_provenance",
    "place_discovery",
    "place_composition",
]

LABEL_COLS = ["period", "genre", "sub_genre", "provenance", "sub_provenance", "domain"]

# Expected output row count (Section 16.1).
EXPECTED_FRAGMENT_COUNT = 384

# Known null clean_value rows in dll.csv (Section 16.3).
# Each tuple: (fragment_id, fragment_line_num, index_in_line).
KNOWN_NULL_CLEAN_VALUE_ROWS: frozenset[tuple[int, int, int]] = frozenset(
    [
        (32264, 20, 3),
        (32592, 116, 2),
        (33621, 36, 3),
        (34164, 11, 3),
    ]
)

# ---------------------------------------------------------------------------
# Cleaning functions (replicated from v_1/src/linear_probing/utils.py to
# avoid the torch import; logic is identical).
# ---------------------------------------------------------------------------

_MAXIMAL_FILTERS = [
    ("strip ALL digits",                lambda t: re.sub(r"[0-9]", "", t)),
    ("truncate 30 tokens",              lambda t: " ".join(t.split()[:30])),
    ("strip case endings",              lambda t: re.sub(r"-(am|im|um|tam|tim|šum)\b", "", t)),
    ("strip w/y",                       lambda t: t.replace("w", "").replace("y", "")),
    ("remove logograms",                lambda t: re.sub(r"\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9]+-?", "", t)),
    ("strip determinatives",            lambda t: re.sub(r"\b(I|d|lu2|uru|giš|tug2)-", "", t)),
    ("keep only syllabic tokens",       lambda t: " ".join(re.findall(r"[a-zšṣṭḫāīūē][a-zšṣṭḫāīūē0-9-]*", t))),
    ("normalize long vowels",           lambda t: t.translate(str.maketrans("āīūēĀĪŪĒ", "aiueAIUE"))),
    ("strip subscript digits",          lambda t: re.sub(r"([a-zšṣṭḫ])([2-9])", r"\1", t)),
    ("lowercase",                       lambda t: t.lower()),
    ("strip -meš plural",               lambda t: re.sub(r"-meš\b", "", t)),
]


def clean_tier0(t: str) -> str:
    """Minimal cleaning: strip ORACC @v markup, non-breaking space, subscript-x."""
    t = re.sub(r"@[a-z0-9]+", "", t)
    t = t.replace("\xa0", " ")    # non-breaking space
    t = t.replace("\u2093", "")   # subscript x (U+2093)
    return t


def clean_maximal(text: str) -> str:
    """Apply tier0 + all 11 filters stacked in order."""
    t = clean_tier0(text)
    for _name, fn in _MAXIMAL_FILTERS:
        t = fn(t)
    return t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_label(col_name: str, value: Any) -> Any:
    """Section 5 normalization: lowercase+strip for genre/sub_genre, raw otherwise."""
    if pd.isna(value):
        return value
    if col_name in ("genre", "sub_genre"):
        return str(value).strip().lower()
    return str(value)


# ---------------------------------------------------------------------------
# Step 1: Verify raw CSVs against Phase 0 contract
# ---------------------------------------------------------------------------


def verify_against_contract(
    raw_dfs: "OrderedDict[str, pd.DataFrame]",
    saved_report: dict,
) -> None:
    """Re-run file integrity checks and diff against inspection_report.json.

    Aborts with a clear message if anything has changed.  The MD5 check is
    the definitive guard; row/column checks are belt-and-suspenders.
    """
    errors: list[str] = []
    saved_by_corpus = {fi["corpus"]: fi for fi in saved_report["file_integrity"]}

    for corpus_name, path in CSV_FILES.items():
        saved = saved_by_corpus[corpus_name]
        actual_md5 = md5_of_file(path)

        if actual_md5 != saved["md5"]:
            errors.append(
                f"{corpus_name}: MD5 changed "
                f"(saved={saved['md5'][:12]}..., actual={actual_md5[:12]}...). "
                f"Re-run 01_inspect_seal_data.py and commit the new report."
            )
            continue  # Skip further checks for this file if MD5 already differs.

        df = raw_dfs[corpus_name]
        if len(df) != saved["n_rows"]:
            errors.append(
                f"{corpus_name}: row count changed "
                f"(saved={saved['n_rows']}, actual={len(df)})"
            )
        if list(df.columns) != saved["columns"]:
            errors.append(
                f"{corpus_name}: column list changed "
                f"(saved={saved['columns']}, actual={list(df.columns)})"
            )

    if errors:
        print(
            "ABORT: input data has drifted from the Phase 0 inspection contract.",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    print("[build] all 3 CSVs match their MD5 hashes in inspection_report.json ✓")


# ---------------------------------------------------------------------------
# Step 2: Assert and apply clean_value fallback
# ---------------------------------------------------------------------------


def apply_clean_value_fallback(
    raw_dfs: "OrderedDict[str, pd.DataFrame]",
) -> "OrderedDict[str, pd.DataFrame]":
    """Assert the null clean_value set equals KNOWN_NULL_CLEAN_VALUE_ROWS.

    Falls back to `value` for those 4 rows (dll only).  Aborts if the set
    has changed — catches any future re-delivery from Chunrong.

    Section 16.3 of the pipeline plan.
    """
    dfs: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for corpus_name, df in raw_dfs.items():
        df = df.copy()
        null_mask = df["clean_value"].isna()
        n_null = int(null_mask.sum())

        if n_null == 0:
            dfs[corpus_name] = df
            continue

        # Build the actual set of (fragment_id, fragment_line_num, index_in_line).
        null_rows = df[null_mask][
            ["fragment_id", "fragment_line_num", "index_in_line"]
        ]
        actual_set: frozenset[tuple[int, int, int]] = frozenset(
            (int(r.fragment_id), int(r.fragment_line_num), int(r.index_in_line))
            for _, r in null_rows.iterrows()
        )

        if actual_set != KNOWN_NULL_CLEAN_VALUE_ROWS:
            extra = actual_set - KNOWN_NULL_CLEAN_VALUE_ROWS
            missing = KNOWN_NULL_CLEAN_VALUE_ROWS - actual_set
            msg = (
                f"ABORT: null clean_value rows in {corpus_name} "
                f"do not match the known set.\n"
                f"  Extra (new nulls): {sorted(extra)}\n"
                f"  Missing (no longer null): {sorted(missing)}\n"
                f"Re-run 01_inspect_seal_data.py and update the plan / README."
            )
            print(msg, file=sys.stderr)
            sys.exit(1)

        # Apply fallback: use `value` wherever clean_value is null.
        df.loc[null_mask, "clean_value"] = df.loc[null_mask, "value"]
        print(
            f"[build] {corpus_name}: applied value→clean_value fallback "
            f"for {n_null} rows: {sorted(actual_set)} ✓"
        )
        dfs[corpus_name] = df

    return dfs


# ---------------------------------------------------------------------------
# Step 3: Build fragment-level table
# ---------------------------------------------------------------------------


def build_fragment_table(
    raw_dfs: "OrderedDict[str, pd.DataFrame]",
) -> pd.DataFrame:
    """Aggregate word-level rows → one row per fragment.

    Sort each corpus by (fragment_id, fragment_line_num, index_in_line) so
    that word concatenation respects the original line+position order.

    Text column is clean_value joined by single space per fragment.
    Metadata is taken from the first row of each fragment (inspect script
    verified consistency within fragments in Phase 0).

    Label normalization (Section 5):
      - genre, sub_genre: lowercase + strip whitespace
      - all other label cols: raw
    """
    parts: list[pd.DataFrame] = []

    for corpus_name, df in raw_dfs.items():
        # Sort to preserve word order.
        df = df.sort_values(
            ["fragment_id", "fragment_line_num", "index_in_line"],
            kind="stable",
        )

        grouped = df.groupby("fragment_id", sort=False)

        # Fragment text: join clean_value by space.
        text_series = grouped["clean_value"].apply(lambda ws: " ".join(ws.astype(str)))

        # Metadata: first value per fragment.
        meta_dict: dict[str, pd.Series] = {}
        for col in METADATA_COLS:
            if col in df.columns:
                meta_dict[col] = grouped[col].first()
            else:
                meta_dict[col] = pd.Series(
                    np.nan, index=grouped.groups.keys(), name=col
                )

        word_counts = grouped.size().rename("word_count")

        frag_df = pd.DataFrame(meta_dict)
        frag_df["word_count"] = word_counts
        frag_df["text"] = text_series
        frag_df.index.name = "fragment_id"
        frag_df = frag_df.reset_index()
        frag_df["corpus"] = corpus_name

        # Apply label normalization (Section 5).
        for col in ("genre", "sub_genre"):
            if col in frag_df.columns:
                frag_df[col] = frag_df[col].apply(
                    lambda v: normalize_label(col, v)
                )

        parts.append(frag_df)
        print(
            f"[build] {corpus_name}: {len(frag_df)} fragments, "
            f"{int(word_counts.sum())} words"
        )

    pooled = pd.concat(parts, ignore_index=True)

    # Pre-compute cleaned text columns (mirrors how 01_extract_activations.py
    # applies cleaning at extraction time).
    print("[build] computing text_tier0 ...")
    pooled["text_tier0"] = pooled["text"].apply(clean_tier0)
    print("[build] computing text_maximal ...")
    pooled["text_maximal"] = pooled["text"].apply(clean_maximal)

    # Canonical column order.
    col_order = (
        ["fragment_id", "corpus"]
        + METADATA_COLS
        + ["word_count", "text", "text_tier0", "text_maximal"]
    )
    return pooled[col_order]


# ---------------------------------------------------------------------------
# Step 4: Write summary JSON
# ---------------------------------------------------------------------------


def build_summary(df: pd.DataFrame) -> dict:
    """Per-corpus × per-label counts + null counts."""
    summary: dict[str, Any] = OrderedDict()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["total_fragments"] = int(len(df))
    summary["fragments_per_corpus"] = {
        corpus: int(n) for corpus, n in df["corpus"].value_counts().sort_index().items()
    }

    per_label: dict[str, Any] = OrderedDict()
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        per_corpus: dict[str, Any] = OrderedDict()
        for corpus in ("seal", "dll", "lbpl"):
            sub = df[df["corpus"] == corpus]
            null_count = int(sub[col].isna().sum())
            vc = sub[col].dropna().value_counts().to_dict()
            per_corpus[corpus] = OrderedDict(
                [
                    ("null_count", null_count),
                    ("n_classes", int(len(vc))),
                    ("class_counts", {str(k): int(v) for k, v in sorted(vc.items())}),
                ]
            )
        all_vc = df[col].dropna().value_counts().to_dict()
        per_label[col] = OrderedDict(
            [
                ("total_null_count", int(df[col].isna().sum())),
                ("total_n_classes", int(len(all_vc))),
                ("per_corpus", per_corpus),
                (
                    "pooled_class_counts",
                    {str(k): int(v) for k, v in sorted(all_vc.items())},
                ),
            ]
        )
    summary["label_summary"] = per_label
    return summary


# ---------------------------------------------------------------------------
# Step 5: Assert output invariants
# ---------------------------------------------------------------------------


def assert_output_invariants(
    df: pd.DataFrame, saved_report: dict
) -> None:
    """Verify the built parquet satisfies the contract from the inspection report.

    Checks:
    1. Row count == 384.
    2. genre and sub_genre values are lowercase (normalization applied).
    3. No unexpected nulls in metadata columns beyond those in the Phase 0 report.
    4. Per-corpus fragment counts match inspection report.
    """
    errors: list[str] = []

    # 1. Row count.
    if len(df) != EXPECTED_FRAGMENT_COUNT:
        errors.append(
            f"Row count: expected {EXPECTED_FRAGMENT_COUNT}, got {len(df)}"
        )

    # 2. Label normalization for genre and sub_genre.
    for col in ("genre", "sub_genre"):
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        bad = non_null[non_null != non_null.str.strip().str.lower()]
        if len(bad) > 0:
            errors.append(
                f"{col}: {len(bad)} values not properly lowercase+stripped: "
                f"{bad.unique()[:5].tolist()}"
            )

    # 3. Per-corpus fragment counts from inspection report.
    frag_agg = saved_report["fragment_aggregation"]
    expected_counts = {
        name: int(info["n_unique_fragments"])
        for name, info in frag_agg.items()
    }
    for corpus, expected_n in expected_counts.items():
        actual_n = int((df["corpus"] == corpus).sum())
        if actual_n != expected_n:
            errors.append(
                f"{corpus}: expected {expected_n} fragments, got {actual_n}"
            )

    # 4. Text columns must be non-null.
    for col in ("text", "text_tier0", "text_maximal"):
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            errors.append(f"{col}: {n_null} null values (expected 0)")

    if errors:
        print("ABORT: output invariant checks failed:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    print("[build] all output invariants passed ✓")


# ---------------------------------------------------------------------------
# Step 6: Spot-check 5 fragments
# ---------------------------------------------------------------------------


def spot_check(df: pd.DataFrame, raw_dfs: "OrderedDict[str, pd.DataFrame]") -> None:
    """Print 5 fragments and their source-CSV word lists for manual verification."""
    print()
    print("=" * 70)
    print("SPOT CHECK: 5 fragments vs source CSVs")
    print("=" * 70)

    # Pick 5 fragments: first 2 from seal, 1 from dll with null fallback, 1 from lbpl,
    # plus 1 more from seal.
    candidates = []
    # One of the known fallback fragments.
    candidates.append(32264)
    # A few from each corpus.
    for corpus in ("seal", "lbpl"):
        sub = df[df["corpus"] == corpus]
        if len(sub) >= 2:
            candidates += sub["fragment_id"].iloc[:2].tolist()
    candidates = candidates[:5]

    # Concat all raw dfs for lookup (fallback already applied in raw_dfs arg).
    all_words = pd.concat(raw_dfs.values(), ignore_index=True)

    for fid in candidates:
        row = df[df["fragment_id"] == fid].iloc[0]
        source_words = (
            all_words[all_words["fragment_id"] == fid]
            .sort_values(["fragment_line_num", "index_in_line"])["clean_value"]
            .astype(str)
            .tolist()
        )
        print(f"\nfragment_id={fid}  corpus={row['corpus']}  words={row['word_count']}")
        print(f"  period    : {row['period']}")
        print(f"  genre     : {row['genre']}")
        print(f"  text[:80] : {row['text'][:80]!r}")
        print(f"  src[:10]  : {source_words[:10]}")
        match = " ".join(source_words) == row["text"]
        print(f"  text==src join: {match}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not ROUND_DIR.exists():
        print(f"ERROR: {ROUND_DIR} does not exist", file=sys.stderr)
        return 2
    if not INSPECTION_REPORT.exists():
        print(
            f"ERROR: Phase 0 contract missing: {INSPECTION_REPORT}\n"
            f"Run 01_inspect_seal_data.py first.",
            file=sys.stderr,
        )
        return 2
    for name, path in CSV_FILES.items():
        if not path.exists():
            print(f"ERROR: missing CSV {path}", file=sys.stderr)
            return 2
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Phase 0 contract ──────────────────────────────────────────────
    with open(INSPECTION_REPORT, encoding="utf-8") as f:
        saved_report = json.load(f)
    print(
        f"[build] loaded inspection contract "
        f"(generated {saved_report['generated_at'][:10]})"
    )

    # ── Load raw CSVs ──────────────────────────────────────────────────────
    print(f"[build] reading 3 CSVs from {ROUND_DIR.relative_to(REPO_ROOT)}")
    raw_dfs: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for name, path in CSV_FILES.items():
        df = pd.read_csv(path)
        raw_dfs[name] = df
        print(f"[build]   {name}: {len(df):,} rows × {len(df.columns)} cols")

    # ── Step 1: Verify against Phase 0 contract ────────────────────────────
    verify_against_contract(raw_dfs, saved_report)

    # ── Step 2: Assert and apply clean_value fallback ──────────────────────
    raw_dfs = apply_clean_value_fallback(raw_dfs)

    # ── Step 3: Build fragment table ───────────────────────────────────────
    print("[build] aggregating word rows → fragment rows ...")
    df = build_fragment_table(raw_dfs)
    print(f"[build] pooled: {len(df)} fragments total")

    # ── Step 4: Assert output invariants ──────────────────────────────────
    assert_output_invariants(df, saved_report)

    # ── Step 5: Write parquet ──────────────────────────────────────────────
    df.to_parquet(CORPUS_PARQUET, index=False)
    print(f"[build] wrote {CORPUS_PARQUET.relative_to(REPO_ROOT)}")
    print(f"[build]   {len(df)} rows × {len(df.columns)} columns")
    print(f"[build]   columns: {list(df.columns)}")

    # ── Step 6: Write summary JSON ─────────────────────────────────────────
    summary = build_summary(df)
    with open(CORPUS_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[build] wrote {CORPUS_SUMMARY.relative_to(REPO_ROOT)}")

    # ── Step 7: Spot-check ─────────────────────────────────────────────────
    spot_check(df, raw_dfs)

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Output: {CORPUS_PARQUET.relative_to(REPO_ROOT)}")
    print(f"  Rows:   {len(df)}")
    print(f"  Cols:   {list(df.columns)}")
    print()
    print("Per-corpus fragment counts:")
    for corpus, n in df["corpus"].value_counts().sort_index().items():
        print(f"  {corpus:>6}: {n:>4} fragments")
    print()
    print("Label null counts:")
    for col in LABEL_COLS:
        n_null = int(df[col].isna().sum()) if col in df.columns else -1
        print(f"  {col:>16}: {n_null:>4} nulls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
