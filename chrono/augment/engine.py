"""View builder: turns corpus rows into augmented views (SLA section 4).

WHAT. build_views expands every corpus row x language x augmentation
chain x seed into one row of the views.parquet schema (view_id, doc_id,
lang, augs, seed, text, n_words, mask_count), with
view_id = "{doc_id}::{lang}::{augs}+s{seed}" and augs the comma-joined
chain ('' = original). sample_view_pair draws one (branch A, branch B)
view pair for JEPA-style training, branch B from the MILDER menu.

WHY PER-VIEW RNG. Each view's randomness is seeded from
(doc_id, lang, augs, seed) alone, so a view is a pure function of its
view_id: rebuilding the table, or re-drawing the same pair, reproduces
byte-identical texts regardless of iteration order or how many other
views were built first.

Ruler spans come from the corpus columns ruler_spans_{eng,akk}
(A1's contract); ops remap them through every edit, so chains like
[crop32, mask_ruler] mask the right characters of the cropped text.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from chrono.augment import ops

VIEW_COLS = ["view_id", "doc_id", "lang", "augs", "seed",
             "text", "n_words", "mask_count"]

# Ordered by the F28 erasure ladder: ruler identity first, then formula
# convention, then object/length nuisances via crops and span drops.
DEFAULT_MENU = [
    [],
    ["mask_ruler"],
    ["strip_formula"],
    ["mask_ruler", "strip_formula"],
    ["mask_ruler", "crop16"],
    ["mask_ruler", "crop32"],
    ["orthonorm"],
    ["mask_ruler", "drop_span"],
]

# Branch B (target) menu: milder by contract.
MENU_MILD = [
    [],
    ["mask_ruler"],
    ["mask_ruler", "crop32"],
]


def _view_rng(doc_id, lang, augs, seed):
    h = hashlib.sha256(f"{doc_id}::{lang}::{augs}".encode()).digest()
    words = [int.from_bytes(h[i:i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng([int(seed)] + words)


def _ruler_spans(doc_row, lang):
    raw = doc_row.get(f"ruler_spans_{lang}")
    if raw is None or (np.isscalar(raw) and pd.isna(raw)):
        return []
    return [[int(a), int(b)] for a, b in raw]


PN_TOKEN = "[PN]"


def _base_text(doc_row, lang, chain):
    """Source text for a view, and its ruler spans.

    REVIEW FIX (wave B1): anglicized royal names never occur in the
    transliteration, so ruler_spans_akk is empty on all 1,187 docs and
    mask_ruler was a byte-for-byte no-op on every Akkadian view — an
    "survives ruler masking" row that passes by construction, plus
    thousands of duplicate texts sent to the GPU. The corpus ships a
    pre-masked Akkadian tier instead ([PN] over personal names), so for a
    chain containing mask_ruler we start from it and rewrite [PN] to the
    typed token. English keeps the span path (46% of glosses name the
    ruler; the rest genuinely never do).
    """
    if lang == "akk" and "mask_ruler" in chain:
        pre = str(doc_row.get("text_akk_masked") or "")
        if pre.strip():
            spans, out, i = [], [], 0
            while True:
                j = pre.find(PN_TOKEN, i)
                if j < 0:
                    out.append(pre[i:])
                    break
                out.append(pre[i:j])
                spans.append([sum(len(x) for x in out), 0])
                out.append(ops.MASK_TOKEN)
                spans[-1][1] = spans[-1][0] + len(ops.MASK_TOKEN)
                i = j + len(PN_TOKEN)
            return "".join(out), {"ruler": spans}
    return (str(doc_row.get(f"text_{lang}") or ""),
            {"ruler": _ruler_spans(doc_row, lang)})


def make_view(doc_row, lang, chain, seed):
    """One view row (dict in VIEW_COLS order) — the single constructor
    both build_views and sample_view_pair go through."""
    augs = ",".join(chain)
    text, spans = _base_text(doc_row, lang, chain)
    rng = _view_rng(doc_row["doc_id"], lang, augs, seed)
    for name in chain:
        if name not in ops.OPS:
            raise KeyError(f"unknown augmentation op: {name!r}")
        text, spans = ops.OPS[name](text, spans, rng)
    return {
        "view_id": f"{doc_row['doc_id']}::{lang}::{augs}+s{seed}",
        "doc_id": doc_row["doc_id"],
        "lang": lang,
        "augs": augs,
        "seed": int(seed),
        "text": text,
        "n_words": len(text.split()),
        "mask_count": text.count(ops.MASK_TOKEN),
    }


def build_views(corpus_df, menu, seeds):
    """Cross product corpus x ('akk','eng') x menu x seeds -> DataFrame
    in the views.parquet schema. Languages with an empty source text are
    skipped for that document."""
    rows = []
    for _, r in corpus_df.iterrows():
        for lang in ("akk", "eng"):
            if not str(r.get(f"text_{lang}") or "").strip():
                continue
            for chain in menu:
                for seed in seeds:
                    rows.append(make_view(r, lang, chain, int(seed)))
    return pd.DataFrame(rows, columns=VIEW_COLS)


def sample_view_pair(doc_row, rng, menu_a, menu_b):
    """Draw one (view_row_a, view_row_b) pair for a document. Branch B
    should be handed the milder menu (e.g. MENU_MILD). All randomness
    (language, chain, per-view seed) comes from `rng`, so an identically
    seeded generator reproduces the pair exactly."""
    langs = [lg for lg in ("akk", "eng")
             if str(doc_row.get(f"text_{lg}") or "").strip()]
    if not langs:
        raise ValueError(f"doc {doc_row['doc_id']!r} has no text")
    out = []
    for menu in (menu_a, menu_b):
        lang = langs[int(rng.integers(len(langs)))]
        chain = menu[int(rng.integers(len(menu)))]
        seed = int(rng.integers(0, 2 ** 31 - 1))
        out.append(make_view(doc_row, lang, chain, seed))
    return out[0], out[1]
