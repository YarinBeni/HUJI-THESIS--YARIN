"""Corpus contract (P0.1): the one canonical fragment table for chrono/.

WHAT. build_corpus() turns the raw ORCC parquet (BC-positive years, three
text tiers, sparse catalogue fields) into the corpus_chrono frame with the
EXACT schema of INTERFACES.md section 3 — one row per eligible fragment,
time converted ONCE to astronomical t (larger = later), ruler-name char
spans precomputed for both languages. build_ruler_table() derives the
reign-proxy table (t_min/t_max from each ruler's own fragments).

WHY. Every sibling module (views, losses, training, eval) consumes this
table and nothing else. Eligibility MUST equal the M.Sc. pipeline's
pairs_data.load_eligible() (year & ruler present, non-empty Akkadian
maximal text, eng_tier0 gloss merged in) so the Phase-0 reproduction gate
compares like with like; the expected census — 1,187 rows / 40 rulers /
47 distinct years — is asserted loudly at build time because silent drift
here would poison every downstream number.

Ruler-name spans reuse the name_variants / span-search approach of
v_1/src/phase2/steering/ignite_anchor.py, extended from first-occurrence
to ALL non-overlapping occurrences and applied to text_eng AND text_akk.
The transliteration almost never contains the anglicized royal name, so
akk span lists are mostly empty — a property of the corpus, not a bug.
text_eng_masked is '' throughout: the raw corpus pre-masks only the
Akkadian tiers (text_maximal_masked); no English pre-masked variant
exists.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from chrono import common

TRANS = os.path.join(common.REPO, "v_1", "src", "stress_tests",
                     "translation", "translations.parquet")

# rows / distinct rulers / distinct years after eligibility filtering
EXPECT = (1187, 40, 47)

COLUMNS = ["doc_id", "ruler", "t", "text_akk", "text_eng",
           "text_akk_masked", "text_eng_masked", "sub_genre",
           "provenance", "period", "n_words",
           "ruler_spans_eng", "ruler_spans_akk"]


def name_variants(ruler: str) -> list:
    """Ruler-name strings to search for, longest first (adapted from
    ignite_anchor.name_variants; the length-then-lexicographic sort key
    replaces set-iteration order so builds are deterministic)."""
    base = str(ruler)
    outs = {base, base.split("(")[0].strip(),
            re.sub(r"\s+[IVX]+$", "", base).strip()}   # drop ordinal
    return sorted({o for o in outs if len(o) >= 4},
                  key=lambda v: (-len(v), v))


def find_spans(text: str, ruler: str) -> list:
    """All non-overlapping char [start, end) ruler-name mentions in text.

    Case-insensitive; longer variants win where variants nest ("Sargon
    II" claims its span before "Sargon" can). Indices always point into
    the ORIGINAL string (regex spans, no lowercased copy)."""
    spans = []
    for v in name_variants(ruler):
        for m in re.finditer(re.escape(v), text, re.IGNORECASE):
            s, e = m.span()
            if all(e <= s0 or s >= e0 for s0, e0 in spans):
                spans.append((s, e))
    return [[s, e] for s, e in sorted(spans)]


def _fill_unk(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return s.mask(s == "", "unk")


def build_corpus(orcc_path=common.ORCC, trans_path=TRANS,
                 strict: bool = True) -> pd.DataFrame:
    """The corpus_chrono frame (INTERFACES.md section 3), sorted by
    doc_id. strict=True (the default) hard-fails unless the census
    matches EXPECT — pass False only for synthetic inputs."""
    df = pd.read_parquet(orcc_path)
    tr = pd.read_parquet(trans_path)[["fragment_id", "eng_tier0"]]
    df = df[df["year"].notna() & df["ruler"].notna()].copy()
    df = df.merge(tr, on="fragment_id", how="left")
    df["text_akk"] = df["text_maximal"].fillna("").astype(str)
    df = df[df["text_akk"].str.strip().str.len() > 0].copy()

    out = pd.DataFrame({
        "doc_id": df["fragment_id"].astype(str),
        "ruler": df["ruler"].astype(str),
        "t": common.to_astro(df["year"].astype(float).values),
        "text_akk": df["text_akk"].values,
        "text_eng": df["eng_tier0"].fillna("").astype(str).values,
        "text_akk_masked":
            df["text_maximal_masked"].fillna("").astype(str).values,
        "text_eng_masked": "",
        "sub_genre": _fill_unk(df["sub_genre"]).values,
        "provenance": _fill_unk(df["provenance"]).values,
        "period": _fill_unk(df["period"]).values,
    })
    out["n_words"] = out["text_eng"].str.split().str.len().astype(int)
    out["ruler_spans_eng"] = [
        find_spans(t, r) for t, r in zip(out["text_eng"], out["ruler"])]
    out["ruler_spans_akk"] = [
        find_spans(t, r) for t, r in zip(out["text_akk"], out["ruler"])]
    out = out[COLUMNS].sort_values("doc_id").reset_index(drop=True)

    if not out["doc_id"].is_unique:
        raise AssertionError("duplicate doc_id in eligible corpus")
    if not (out["t"] < 0).all():
        raise AssertionError("non-negative t: BC-positive year leaked in")
    got = (len(out), out["ruler"].nunique(), out["t"].nunique())
    if strict and got != EXPECT:
        raise AssertionError(
            f"corpus census drifted: got {got} (rows/rulers/years), "
            f"expected {EXPECT} — eligibility no longer mirrors "
            "pairs_data.load_eligible(); do NOT ship this artifact")
    return out


def build_ruler_table(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Reign PROXY per ruler: [t_min, t_max] spanned by that ruler's own
    fragments (astronomical), not attested regnal dates — hence the
    proxy=True flag. n_docs = fragment count."""
    g = corpus_df.groupby("ruler")["t"]
    return pd.DataFrame({
        "ruler": g.min().index.astype(str),
        "t_min": g.min().values.astype(float),
        "t_max": g.max().values.astype(float),
        "proxy": True,
        "n_docs": g.size().values.astype(int),
    }).reset_index(drop=True)
