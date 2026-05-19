#!/usr/bin/env python
"""03_build_orcc_corpus.py — Build fragment-level ORCC corpus parquet.

Reads:
  v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv

Writes:
  v_1/data/evaluation/corpora/orcc_corpus.parquet

Modelled on 02_build_seal_corpus.py; same output schema plus year and ruler.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = (
    REPO_ROOT / "v_1" / "data" / "raw" / "chungrong" / "orcc_round1"
    / "royal_inscriptions.csv"
)
CORPORA_DIR = REPO_ROOT / "v_1" / "data" / "evaluation" / "corpora"
CORPUS_PARQUET = CORPORA_DIR / "orcc_corpus.parquet"

# ---------------------------------------------------------------------------
# Cleaning functions (copied from 02_build_seal_corpus.py — do not import)
# ---------------------------------------------------------------------------

_MAXIMAL_FILTERS = [
    ("strip ALL digits",            lambda t: re.sub(r"[0-9]", "", t)),
    ("truncate 30 tokens",          lambda t: " ".join(t.split()[:30])),
    ("strip case endings",          lambda t: re.sub(r"-(am|im|um|tam|tim|šum)\b", "", t)),
    ("strip w/y",                   lambda t: t.replace("w", "").replace("y", "")),
    ("remove logograms",            lambda t: re.sub(r"\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9]+-?", "", t)),
    ("strip determinatives",        lambda t: re.sub(r"\b(I|d|lu2|uru|giš|tug2)-", "", t)),
    ("keep only syllabic tokens",   lambda t: " ".join(re.findall(r"[a-zšṣṭḫāīūē][a-zšṣṭḫāīūē0-9-]*", t))),
    ("normalize long vowels",       lambda t: t.translate(str.maketrans("āīūēĀĪŪĒ", "aiueAIUE"))),
    ("strip subscript digits",      lambda t: re.sub(r"([a-zšṣṭḫ])([2-9])", r"\1", t)),
    ("lowercase",                   lambda t: t.lower()),
    ("strip -meš plural",           lambda t: re.sub(r"-meš\b", "", t)),
]


def apply_tier0(t: str) -> str:
    t = re.sub(r"@[a-z0-9]+", "", t)
    t = t.replace("\xa0", " ")
    t = t.replace("ₓ", "")
    return t


def apply_maximal(text: str) -> str:
    t = apply_tier0(text)
    for _name, fn in _MAXIMAL_FILTERS:
        t = fn(t)
    return t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reign end-year fallback for rulers whose sub_period is absent from ORACC.
# Convention: use the end (most recent) year of the reign, matching the min(digits)
# convention applied to sub_period strings (e.g. "ca. 668-631" → 631).
#
# NB rulers (Nebuchadnezzar II, Nabonidus): ORACC/RINBE does not include sub_period
# for these texts, and per-inscription dates are unavailable. Confirmed by Chung-rong
# Ni (email "Adding the corpora of Royal Inscriptions", 2026-05-19): "Oracc did not
# include these information … Oracc does not have more granular per-inscription dates."
# We fill with the well-attested reign end-year rather than leaving NaN, so year
# regression can treat NB rulers on the same footing as NA rulers.
#
# NA rulers (Sargon II, Tiglath-pileser III): a minority of fragments lack sub_period
# in ORACC (same root cause). Fallback is consistent with the non-NaN values those
# rulers already have in the corpus.
_RULER_YEAR_FALLBACK: dict[str, int] = {
    "Nebuchadnezzar II": 562,    # reign 605–562 BCE
    "Nabonidus":         539,    # reign 556–539 BCE
    "Sargon II":         705,    # reign 722–705 BCE (matches existing sub_period values)
    "Tiglath-pileser III": 727,  # reign 745–727 BCE (matches existing sub_period values)
}


def extract_year(sub_period) -> int | None:
    if pd.isna(sub_period):
        return None
    digits = re.findall(r'\d+', str(sub_period))
    if not digits:
        return None
    return min(int(d) for d in digits)


def extract_year_with_fallback(sub_period, ruler: str | None) -> int | None:
    year = extract_year(sub_period)
    if year is None and ruler in _RULER_YEAR_FALLBACK:
        return _RULER_YEAR_FALLBACK[ruler]
    return year


def extract_ruler(domain) -> str | None:
    """Return content of first parenthetical, or the domain value as-is.

    Examples: 'ribo(Esarhaddon)' → 'Esarhaddon', 'ribo' → 'ribo'
    """
    if pd.isna(domain):
        return None
    m = re.search(r'\((.+?)\)', str(domain))
    if m:
        return m.group(1)
    return str(domain)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} does not exist", file=sys.stderr)
        return 2
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {CSV_PATH.name}")

    # Apply clean_value fallback: use value when clean_value is null
    null_mask = df["clean_value"].isna()
    n_null = int(null_mask.sum())
    if n_null > 0:
        df.loc[null_mask, "clean_value"] = df.loc[null_mask, "value"]
        print(f"Applied value→clean_value fallback for {n_null} rows")

    # Sort for consistent word order
    df = df.sort_values(
        ["fragment_id", "fragment_line_num", "index_in_line"], kind="stable"
    )

    # Aggregate to fragment level
    grouped = df.groupby("fragment_id", sort=False)

    text_series = grouped["clean_value"].apply(lambda ws: " ".join(ws.dropna().astype(str)))
    word_counts = grouped.size().rename("word_count")

    meta_cols = [
        "word_language", "domain", "period", "sub_period",
        "genre", "sub_genre", "provenance", "sub_provenance",
    ]
    meta = {col: grouped[col].first() for col in meta_cols if col in df.columns}

    frag_df = pd.DataFrame(meta)
    frag_df["word_count"] = word_counts
    frag_df["text"] = text_series
    frag_df.index.name = "fragment_id"
    frag_df = frag_df.reset_index()

    # Compute ruler before overwriting domain
    frag_df["ruler"] = frag_df["domain"].apply(extract_ruler)

    # Compute year from sub_period, with per-ruler fallback for rulers whose
    # sub_period is absent from ORACC (see _RULER_YEAR_FALLBACK — this IS the
    # permanent fix; no upstream data update expected).
    if "sub_period" in frag_df.columns:
        years = frag_df.apply(
            lambda r: extract_year_with_fallback(r["sub_period"], r.get("ruler")),
            axis=1,
        )
        frag_df["year"] = years.astype("Int64")
    else:
        frag_df["year"] = pd.array([None] * len(frag_df), dtype=pd.Int64Dtype())

    # Normalize corpus and domain fields
    frag_df["corpus"] = "orcc"
    frag_df["domain"] = "ORCC"

    # Compute cleaned text columns
    print("Computing text_tier0 ...")
    frag_df["text_tier0"] = frag_df["text"].apply(apply_tier0)
    print("Computing text_maximal ...")
    frag_df["text_maximal"] = frag_df["text"].apply(apply_maximal)

    # Canonical column order
    col_order = [
        "fragment_id", "corpus", "word_language", "domain", "ruler",
        "period", "genre", "sub_genre", "provenance", "sub_provenance",
        "word_count", "text", "text_tier0", "text_maximal", "year",
    ]
    col_order = [c for c in col_order if c in frag_df.columns]
    frag_df = frag_df[col_order]

    # Save
    frag_df.to_parquet(CORPUS_PARQUET, index=False)
    print(f"\nWrote {CORPUS_PARQUET.relative_to(REPO_ROOT)}")
    print(f"  {len(frag_df)} rows × {len(frag_df.columns)} columns")
    print(f"  columns: {list(frag_df.columns)}")

    # Summary
    n_year = int(frag_df["year"].notna().sum())
    n_no_year = int(frag_df["year"].isna().sum())
    print(f"\nTotal fragments: {len(frag_df)}")
    print(f"Fragments with year: {n_year}  |  without year: {n_no_year}")
    if n_year > 0:
        y_min = int(frag_df["year"].dropna().min())
        y_max = int(frag_df["year"].dropna().max())
        print(f"Year range: {y_min} – {y_max}")
    print("\nPeriod value_counts:")
    print(frag_df["period"].value_counts().to_string())
    print("\nTop 10 ruler value_counts:")
    print(frag_df["ruler"].value_counts().head(10).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
