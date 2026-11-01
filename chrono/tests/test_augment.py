"""A2 tests: augmentation ops, view engine, formula rules, confound audit.

Everything runs on the toy_corpus fixture except the strip_formula
precision check, which reads five real tier-0 texts straight from the
ORCC parquet (skipped cleanly when that file is absent, e.g. on a bare
checkout).
"""
import os

import numpy as np
import pandas as pd
import pytest

from chrono import common
from chrono.augment import audit, engine, formulae, ops


def _rng():
    return np.random.default_rng(0)


def test_mask_ruler_idempotent(toy_corpus):
    row = toy_corpus.iloc[0]
    spans = {"ruler": [list(s) for s in row.ruler_spans_eng]}
    t1, s1 = ops.mask_ruler(row.text_eng, spans, _rng())
    t2, s2 = ops.mask_ruler(t1, s1, _rng())
    assert t1.startswith(ops.MASK_TOKEN + " king of Assyria")
    assert s1["ruler"] == [[0, len(ops.MASK_TOKEN)]]
    assert (t2, s2) == (t1, s1)


def test_mask_ruler_no_mention_unchanged():
    text = "the palace wall was restored and the canal dug"
    for spans in ({}, {"ruler": []}):
        out, out_spans = ops.mask_ruler(text, spans, _rng())
        assert out == text
        assert out_spans == spans


def test_mask_ruler_pure(toy_corpus):
    row = toy_corpus.iloc[0]
    spans = {"ruler": [list(s) for s in row.ruler_spans_eng]}
    before = {k: [list(s) for s in v] for k, v in spans.items()}
    ops.mask_ruler(row.text_eng, spans, _rng())
    assert spans == before          # inputs never mutated


def test_crops_exact_word_counts(toy_corpus):
    row = toy_corpus.iloc[0]
    n_all = len(row.text_eng.split())
    spans = {"ruler": [list(s) for s in row.ruler_spans_eng]}
    for n, op in [(8, ops.crop8), (16, ops.crop16), (32, ops.crop32)]:
        assert n < n_all
        out, out_spans = op(row.text_eng, spans, _rng())
        assert len(out.split()) == n
        for a, b in out_spans["ruler"]:     # remapped spans stay in text
            assert 0 <= a < b <= len(out)
    big, _ = ops.crop64(row.text_eng, spans, _rng())
    assert big == row.text_eng              # doc shorter than the window


def test_default_menu_six_distinct_valid_views(toy_corpus):
    views = engine.build_views(toy_corpus, engine.DEFAULT_MENU, [0])
    assert list(views.columns) == engine.VIEW_COLS
    assert (views.n_words >= 5).all()
    assert (views.text.str.len() > 0).all()
    distinct = views.groupby(["doc_id", "lang"])["text"].nunique()
    assert len(distinct) == 2 * len(toy_corpus)
    assert (distinct >= 6).all()


def test_view_id_deterministic(toy_corpus):
    a = engine.build_views(toy_corpus, engine.DEFAULT_MENU, [0, 1])
    b = engine.build_views(toy_corpus, engine.DEFAULT_MENU, [0, 1])
    pd.testing.assert_frame_equal(a, b)
    assert not a.view_id.duplicated().any()
    row = a.iloc[len(engine.DEFAULT_MENU)]  # doc 0, akk, chain 1, seed 0
    assert row.view_id == f"{row.doc_id}::{row.lang}::{row.augs}+s0"


def test_sample_view_pair_reproducible(toy_corpus):
    row = toy_corpus.iloc[3]
    a1, b1 = engine.sample_view_pair(row, common.rng(7),
                                     engine.DEFAULT_MENU, engine.MENU_MILD)
    a2, b2 = engine.sample_view_pair(row, common.rng(7),
                                     engine.DEFAULT_MENU, engine.MENU_MILD)
    assert (a1, b1) == (a2, b2)
    for v in (a1, b1):
        assert sorted(v) == sorted(engine.VIEW_COLS)
        assert v["doc_id"] == row.doc_id
    assert ",".join([c for m in engine.MENU_MILD for c in m]).find(
        b1["augs"].split(",")[0]) >= 0 or b1["augs"] == ""


def test_strip_formula_min_words_guard():
    text = "king of Assyria great king"       # all formula, 5 words
    out, spans = ops.strip_formula(text, {"ruler": []}, _rng())
    assert out == text
    _, removed, flagged = formulae.strip_formulae(text)
    assert flagged and removed


@pytest.mark.skipif(not os.path.exists(common.ORCC),
                    reason="real ORCC parquet not present")
def test_strip_formula_on_real_gloss_texts():
    df = pd.read_parquet(common.ORCC, columns=["text_tier0"])
    texts = []
    for t in df.text_tier0.astype(str):
        if len(t.split()) >= 12 and formulae.find_formula_spans(t):
            texts.append(t)
        if len(texts) == 5:
            break
    assert len(texts) == 5
    for t in texts:
        before = formulae.find_formula_spans(t)
        out, _ = ops.strip_formula(t, {"ruler": []}, _rng())
        assert len(out.split()) >= formulae.MIN_WORDS_RETAINED
        assert len(out.split()) < len(t.split())
        assert len(formulae.find_formula_spans(out)) < len(before)


def test_audit_joins_one_to_one(toy_corpus):
    views = engine.build_views(toy_corpus, engine.DEFAULT_MENU, [0])
    table = audit.confound_table(views, toy_corpus)
    assert list(table.columns) == audit.COLS
    assert len(table) == len(views)
    assert not table.view_id.duplicated().any()
    assert set(table.view_id) == set(views.view_id)
    joined = views[["view_id", "doc_id", "augs", "n_words"]].merge(
        table, on="view_id", validate="1:1")
    assert (joined.length == joined.n_words).all()
    meta = toy_corpus.set_index("doc_id")
    assert (joined.sub_genre.values ==
            meta.loc[joined.doc_id, "sub_genre"].values).all()
    # a crop after mask_ruler may drop the token; the un-cropped masked
    # view must carry it
    masked = joined[joined.augs == "mask_ruler"]
    assert len(masked) and (masked.mask_count >= 1).all()
