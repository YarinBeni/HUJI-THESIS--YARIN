"""Confound audit: one nuisance-variable row per view (SLA section 4).

WHY. The HSIC penalty and the leakage probes need, for every view, the
nuisance variables a lateness score must NOT depend on: view length,
how many ruler masks it carries, and the document's object type and
find-spot. This table is the single place those are materialized, keyed
by view_id so it joins 1:1 with views.parquet.
"""
from __future__ import annotations

import os

import pandas as pd

from chrono import common

COLS = ["view_id", "length", "mask_count", "sub_genre", "provenance"]


def confound_table(views_df, corpus_df=None):
    """views_df (views.parquet schema) -> DataFrame with COLS, one row
    per view. sub_genre/provenance come from corpus_df when given, else
    from artifacts/corpus_chrono.parquet when it exists, else 'unk'."""
    if corpus_df is None:
        p = os.path.join(common.ART, "corpus_chrono.parquet")
        if os.path.exists(p):
            corpus_df = pd.read_parquet(p)
    out = views_df[["view_id", "doc_id", "mask_count"]].copy()
    out["length"] = views_df["n_words"].astype(int).values
    if corpus_df is not None:
        meta = corpus_df[["doc_id", "sub_genre", "provenance"]]
        out = out.merge(meta.drop_duplicates("doc_id"),
                        on="doc_id", how="left")
    for c in ("sub_genre", "provenance"):
        if c not in out:
            out[c] = "unk"
        out[c] = out[c].fillna("unk")
    return out[COLS]
