"""A1 invariants on the REAL data artifacts (P0.1/P0.2).

Runs against chrono/artifacts/{corpus_chrono.parquet, ruler_table.parquet,
splits/*.json}; artifacts/ is gitignored, so missing files are rebuilt by
running the actual scripts — which is itself part of the test surface.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from chrono import common
from chrono.data import contract, splits

CORPUS_P = os.path.join(common.ART, "corpus_chrono.parquet")
RULER_P = os.path.join(common.ART, "ruler_table.parquet")
SPLITS_DIR = os.path.join(common.ART, "splits")
SCRIPTS = os.path.join(common.CHRONO, "scripts")


def _ensure_artifacts():
    if not (os.path.exists(CORPUS_P) and os.path.exists(RULER_P)):
        subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "make_corpus.py")],
            check=True)
    if any(not os.path.exists(os.path.join(SPLITS_DIR, n + ".json"))
           for n in splits.SPLIT_NAMES):
        subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "make_splits.py")],
            check=True)


@pytest.fixture(scope="module")
def corpus():
    _ensure_artifacts()
    return pd.read_parquet(CORPUS_P)


@pytest.fixture(scope="module")
def ruler_table():
    _ensure_artifacts()
    return pd.read_parquet(RULER_P)


@pytest.fixture(scope="module")
def split_files():
    """name -> (raw file bytes, parsed dict) for all five splits."""
    _ensure_artifacts()
    out = {}
    for n in splits.SPLIT_NAMES:
        with open(os.path.join(SPLITS_DIR, n + ".json"), "rb") as f:
            raw = f.read()
        out[n] = (raw, json.loads(raw))
    return out


# ---------------------------------------------------------------- corpus

def test_census(corpus):
    assert len(corpus) == 1187
    assert corpus["ruler"].nunique() == 40
    assert corpus["t"].nunique() == 47


def test_schema(corpus):
    assert list(corpus.columns) == contract.COLUMNS
    assert corpus["doc_id"].is_unique
    assert corpus["t"].dtype == np.float64
    assert np.issubdtype(corpus["n_words"].dtype, np.integer)
    for c in ("text_akk", "text_eng", "text_akk_masked",
              "text_eng_masked"):
        assert corpus[c].map(lambda v: isinstance(v, str)).all()


def test_t_strictly_negative(corpus):
    assert (corpus["t"] < 0).all()


def test_categoricals_unk_filled(corpus):
    for c in ("sub_genre", "provenance", "period"):
        col = corpus[c]
        assert col.notna().all()
        assert (col.astype(str).str.strip() != "").all()


def test_n_words_is_eng_whitespace_tokens(corpus):
    calc = corpus["text_eng"].str.split().str.len()
    assert (corpus["n_words"] == calc).all()


def test_spans_in_bounds_and_matching(corpus):
    n_eng_hit = 0
    for row in corpus.itertuples(index=False):
        for lang, text in (("eng", row.text_eng), ("akk", row.text_akk)):
            variants = {v.lower()
                        for v in contract.name_variants(row.ruler)}
            got = sorted((int(s), int(e))
                         for s, e in getattr(row, f"ruler_spans_{lang}"))
            prev_end = 0
            for s, e in got:
                assert 0 <= s < e <= len(text)
                assert s >= prev_end          # non-overlapping
                prev_end = e
                sub = text[s:e].lower()
                assert (sub in variants
                        or str(row.ruler).lower().startswith(sub))
        n_eng_hit += len(row.ruler_spans_eng) > 0
    # ignite_anchor measured ~46% of glosses carrying the royal name
    assert n_eng_hit / len(corpus) > 0.35


# ----------------------------------------------------------- ruler table

def test_ruler_table(ruler_table, corpus):
    assert list(ruler_table.columns) == ["ruler", "t_min", "t_max",
                                         "proxy", "n_docs"]
    assert len(ruler_table) == 40
    assert (ruler_table["t_min"] <= ruler_table["t_max"]).all()
    assert ruler_table["proxy"].dtype == bool
    assert ruler_table["proxy"].all()
    assert ruler_table["n_docs"].sum() == len(corpus)
    g = corpus.groupby("ruler")["t"]
    rt = ruler_table.set_index("ruler")
    assert (rt["t_min"] == g.min()).all()
    assert (rt["t_max"] == g.max()).all()
    assert (rt["n_docs"] == g.size()).all()


# ---------------------------------------------------------------- splits

def test_split_schema_and_disjointness(split_files, corpus):
    ids = set(corpus["doc_id"])
    for name, (_, sp) in split_files.items():
        assert set(sp) == {"name", "kind", "seed", "folds"}
        assert sp["name"] == name
        assert isinstance(sp["seed"], int)
        assert len(sp["folds"]) >= 1
        for f in sp["folds"]:
            assert set(f) == {"train", "test"}
            tr, te = f["train"], f["test"]
            assert tr == sorted(tr) and te == sorted(te)
            assert len(set(tr)) == len(tr) and len(set(te)) == len(te)
            assert not set(tr) & set(te)
            assert set(tr) <= ids and set(te) <= ids


def test_gkf_ruler(split_files, corpus):
    sp = split_files["gkf_ruler"][1]
    assert len(sp["folds"]) == 5
    by_doc = dict(zip(corpus["doc_id"], corpus["ruler"]))
    seen = []
    for f in sp["folds"]:
        te_rulers = {by_doc[d] for d in f["test"]}
        tr_rulers = {by_doc[d] for d in f["train"]}
        assert not te_rulers & tr_rulers   # no ruler straddles the fold
        assert len(f["train"]) + len(f["test"]) == len(corpus)
        seen += f["test"]
    assert sorted(seen) == sorted(corpus["doc_id"])   # a partition


def test_mc_balanced_8x21(split_files, corpus):
    sp = split_files["mc_balanced"][1]
    assert len(sp["folds"]) == 200
    by_doc = dict(zip(corpus["doc_id"], corpus["ruler"]))
    for f in sp["folds"]:
        assert len(f["test"]) == 168
        vc = pd.Series([by_doc[d] for d in f["test"]]).value_counts()
        assert len(vc) == 8
        assert (vc == 21).all()
        assert len(f["train"]) == len(corpus) - 168


def test_loro(split_files, corpus):
    sp = split_files["loro"][1]
    counts = corpus["ruler"].value_counts()
    expected = sorted(counts.index[counts >= 10])
    assert len(sp["folds"]) == len(expected)
    by_doc = dict(zip(corpus["doc_id"], corpus["ruler"]))
    held = []
    for f in sp["folds"]:
        rulers = {by_doc[d] for d in f["test"]}
        assert len(rulers) == 1
        r = rulers.pop()
        assert len(f["test"]) == counts[r] >= 10
        held.append(r)
    assert sorted(held) == expected


@pytest.mark.parametrize("name,col", [("source_held_out", "provenance"),
                                      ("object_held_out", "sub_genre")])
def test_category_held_out(split_files, corpus, name, col):
    sp = split_files[name][1]
    assert len(sp["folds"]) == 5
    counts = corpus.loc[corpus[col] != "unk", col].value_counts()
    top5 = sorted(counts.index, key=lambda v: (-counts[v], v))[:5]
    by_doc = dict(zip(corpus["doc_id"], corpus[col]))
    held = []
    for f in sp["folds"]:
        vals = {by_doc[d] for d in f["test"]}
        assert len(vals) == 1
        v = vals.pop()
        assert v != "unk"
        assert len(f["test"]) == counts[v]   # ALL docs of v held out
        held.append(v)
    assert sorted(held) == sorted(top5)


def test_splits_byte_identical(split_files, corpus):
    """Two independent rebuilds are byte-identical to each other AND to
    the frozen artifacts on disk."""
    seed = split_files["gkf_ruler"][1]["seed"]
    run1 = splits.build_all(corpus, seed=seed)
    run2 = splits.build_all(corpus, seed=seed)
    for name, (raw, _) in split_files.items():
        b1 = splits.to_json_bytes(run1[name])
        b2 = splits.to_json_bytes(run2[name])
        assert b1 == b2, f"{name}: rebuild not deterministic"
        assert b1 == raw, f"{name}: frozen artifact drifted from code"
