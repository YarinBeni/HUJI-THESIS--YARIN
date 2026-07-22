"""Akkadian data for the Gurnee & Tegmark mimic (WA section).

We apply G&T's *headline* protocol to Akkadian: the entity string is a whole text
fragment (its maximal-cleaned Akkadian, or its English translation), and we probe the
last-token embedding for the fragment's YEAR (their historical/headline analog) and
for its find-spot's (lat, lon) (their world_place analog).

Sources (all already in the repo):
  * corpus:       v_1/data/evaluation/corpora/orcc_corpus.parquet
                  (fragment_id, ruler, year, provenance, text_maximal, ...)
  * translations: v_1/src/stress_tests/translation/translations.parquet
                  (fragment_id, eng_tier0, eng_maximal)
  * gazetteer:    v_1/src/stress_tests/shared/sites_gazetteer.csv
                  (provenance -> lat, lon)

Two ruler sets (both requested):
  * r8  — the 8 best-attested rulers (>=20 dated texts): the clean, dense subset.
  * r40 — all rulers with a year label (the full, sparser tail).

Two text variants:
  * akk_maximal — the Akkadian text (text_maximal)
  * eng_maximal — its English translation (eng_maximal)   [the translation probe]
"""
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # repo root
CORPUS = os.path.join(_ROOT, "v_1/data/evaluation/corpora/orcc_corpus.parquet")
TRANS = os.path.join(_ROOT, "v_1/src/stress_tests/translation/translations.parquet")
GAZ = os.path.join(_ROOT, "v_1/src/stress_tests/shared/sites_gazetteer.csv")

TEXT_VARIANTS = {"akk_maximal": "text_akk", "eng_maximal": "text_eng"}
RULER_SETS = ["r8", "r40"]
TARGETS = ["year", "geo"]          # geo = (lon, lat), like G&T world_place
N_R8 = 8
TEST_RATIO = 0.2
SEED = 42


def load_fragments() -> pd.DataFrame:
    """One row per dated fragment with both text variants and, where known, coords.
    Columns: fragment_id, ruler, year, lon, lat, has_geo, text_akk, text_eng."""
    df = pd.read_parquet(CORPUS)[
        ["fragment_id", "ruler", "year", "provenance", "text_maximal"]]
    tr = pd.read_parquet(TRANS)[["fragment_id", "eng_maximal"]]
    gaz = pd.read_csv(GAZ)[["provenance", "lat", "lon"]]

    df = df[df.year.notna() & df.ruler.notna()].copy()
    df = df.merge(tr, on="fragment_id", how="left")
    df = df.merge(gaz, on="provenance", how="left")

    df["text_akk"] = df["text_maximal"].fillna("").astype(str)
    df["text_eng"] = df["eng_maximal"].fillna("").astype(str)
    df["year"] = df["year"].astype(float)
    df["has_geo"] = df["lat"].notna() & df["lon"].notna()
    # drop rows whose Akkadian text is empty (nothing to embed)
    df = df[df["text_akk"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


def ruler_set_mask(df: pd.DataFrame, ruler_set: str) -> np.ndarray:
    """Boolean mask selecting the fragments of a ruler set."""
    if ruler_set == "r40":
        return np.ones(len(df), dtype=bool)
    if ruler_set == "r8":
        top = df.ruler.value_counts().head(N_R8).index
        return df.ruler.isin(top).values
    raise ValueError(f"unknown ruler set {ruler_set!r}")


def r8_rulers(df: pd.DataFrame) -> list:
    return list(df.ruler.value_counts().head(N_R8).index)


def entity_texts(df: pd.DataFrame, variant: str) -> list:
    """The entity strings the model sees for a text variant (whole fragment)."""
    col = TEXT_VARIANTS[variant]
    return [str(t) for t in df[col].values]


def target_values(df: pd.DataFrame, target: str):
    """Returns (target array, valid mask). year -> (n,) float; geo -> (n,2) [lon,lat]
    over rows with known coords (valid mask marks the rest for exclusion)."""
    if target == "year":
        return df["year"].values.astype(float), np.ones(len(df), dtype=bool)
    if target == "geo":
        t = df[["lon", "lat"]].values.astype(float)
        return t, df["has_geo"].values
    raise ValueError(f"unknown target {target!r}")


def is_test_split(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Stratified-by-ruler random hold-out (seed 42) over the selected fragments, so
    every ruler appears in train and test — the held-out-entity split, G&T-style.
    Returns a boolean is_test array aligned to df (False outside the mask)."""
    rng = np.random.RandomState(SEED)
    is_test = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(mask)
    for ruler, grp in df.iloc[idx].groupby("ruler"):
        rows = grp.index.values
        k = max(1, int(round(len(rows) * TEST_RATIO))) if len(rows) > 1 else 0
        if k:
            is_test[rng.choice(rows, size=k, replace=False)] = True
    return is_test
