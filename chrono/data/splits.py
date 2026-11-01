"""Split factory (P0.2): the five frozen evaluation splits of chrono/.

WHAT. One builder per split file of INTERFACES.md section 3 — GroupKFold
by ruler, 200 balanced Monte-Carlo draws (8 rulers x 21 docs), leave-one-
ruler-out, source-held-out, object-held-out — plus the canonical JSON
serializer. Every builder sorts its inputs and ids, so a (corpus, seed)
pair maps to byte-identical JSON on every run and every machine.

WHY frozen. Splits are the leak-control of the whole program: the M.Sc.
result only survived under GroupKFold-by-ruler, and every chrono claim
must be evaluated on splits fixed BEFORE training. Freezing them as
byte-stable artifacts makes "same split" checkable with a file hash.

Schema (exact): {"name": str, "kind": str, "seed": int,
"folds": [{"train": [doc_id], "test": [doc_id]}]} — sorted keys, sorted
id lists. mc_balanced folds are evaluation draws, not generalization
splits: "test" is the 168 balanced docs, "train" the eligible remainder
(same-ruler docs on both sides, by design). The 'unk' fill value never
qualifies as a held-out category — holding out "unknown" tests nothing.


REVIEW FIX (wave B1) — how to read mc_balanced. Exactly 8 rulers have
>= 21 docs, so every one of the 200 draws samples the SAME 8 rulers; one
ruler with exactly 21 docs contributes all of them to every draw, mean
pairwise test overlap is ~48/168 docs and 116 eligible docs never appear
in any draw. The draws are therefore heavily dependent resamples of ONE
fixed 8-ruler design: the spread of per-draw rho is doc-resampling noise,
NOT a standard error, and must never be divided by sqrt(200). Any
model-vs-model claim needs ruler-level uncertainty (leave-one-ruler-out
deltas or a bootstrap over the 8 rulers) plus the block placebo in
chrono.eval.block_placebo_rho.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from chrono import common

SEED = 42
N_FOLDS = 5
N_DRAWS = 200
N_RULERS = 8
PER_RULER = 21          # echoes the regression design's k=21 cap
LORO_MIN_DOCS = 10
TOP_K = 5

SPLIT_NAMES = ["gkf_ruler", "mc_balanced", "loro",
               "source_held_out", "object_held_out"]


def _fold(train_ids, test_ids) -> dict:
    return {"train": sorted(str(i) for i in train_ids),
            "test": sorted(str(i) for i in test_ids)}


def _split(name, kind, seed, folds) -> dict:
    return {"name": name, "kind": kind, "seed": int(seed), "folds": folds}


def _sorted_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    return corpus_df.sort_values("doc_id").reset_index(drop=True)


def build_gkf_ruler(corpus_df, n_folds: int = N_FOLDS,
                    seed: int = SEED) -> dict:
    """GroupKFold by ruler: no ruler straddles train and test — the
    protocol that killed the identity leak in the regression design.
    sklearn's assignment is deterministic; seed is recorded provenance."""
    from sklearn.model_selection import GroupKFold
    df = _sorted_corpus(corpus_df)
    ids, groups = df["doc_id"].values, df["ruler"].values
    folds = [_fold(ids[tr], ids[te]) for tr, te in
             GroupKFold(n_splits=n_folds).split(ids, groups=groups)]
    return _split("gkf_ruler", "group_kfold_by_ruler", seed, folds)


def build_mc_balanced(corpus_df, n_draws: int = N_DRAWS,
                      n_rulers: int = N_RULERS,
                      per_ruler: int = PER_RULER,
                      seed: int = SEED) -> dict:
    """n_draws balanced draws: sample n_rulers rulers (only those with
    >= per_ruler docs are eligible for a draw), then per_ruler docs from
    each without replacement. "test" is the balanced sample the metric
    runs on; "train" is every other eligible doc."""
    df = _sorted_corpus(corpus_df)
    by_ruler = {r: sorted(g["doc_id"]) for r, g in df.groupby("ruler")}
    eligible = np.array(sorted(r for r, ids in by_ruler.items()
                               if len(ids) >= per_ruler))
    if len(eligible) < n_rulers:
        raise ValueError(
            f"only {len(eligible)} rulers have >= {per_ruler} docs; "
            f"cannot draw {n_rulers}")
    rng = np.random.default_rng(seed)
    all_ids = set(df["doc_id"])
    folds = []
    for _ in range(n_draws):
        pick = rng.choice(eligible, size=n_rulers, replace=False)
        test = []
        for r in sorted(pick):
            test.extend(rng.choice(by_ruler[r], size=per_ruler,
                                   replace=False))
        folds.append(_fold(all_ids - set(test), test))
    return _split("mc_balanced", "monte_carlo_balanced", seed, folds)


def build_loro(corpus_df, min_docs: int = LORO_MIN_DOCS,
               seed: int = SEED) -> dict:
    """Leave-one-ruler-out over rulers with >= min_docs fragments; one
    fold per held-out ruler, in sorted ruler order."""
    df = _sorted_corpus(corpus_df)
    counts = df["ruler"].value_counts()
    rulers = sorted(counts.index[counts >= min_docs])
    all_ids = set(df["doc_id"])
    folds = []
    for r in rulers:
        test = df.loc[df["ruler"] == r, "doc_id"]
        folds.append(_fold(all_ids - set(test), test))
    return _split("loro", "leave_one_ruler_out", seed, folds)


def _top_values(series: pd.Series, k: int) -> list:
    """Top-k category values by count (ties broken alphabetically);
    the 'unk' fill value never qualifies."""
    counts = series[series != "unk"].value_counts()
    return sorted(counts.index, key=lambda v: (-counts[v], v))[:k]


def _category_split(corpus_df, col, name, top_k, seed) -> dict:
    df = _sorted_corpus(corpus_df)
    all_ids = set(df["doc_id"])
    folds = []
    for v in _top_values(df[col], top_k):
        test = df.loc[df[col] == v, "doc_id"]
        folds.append(_fold(all_ids - set(test), test))
    return _split(name, f"held_out_{col}", seed, folds)


def build_source_held_out(corpus_df, top_k: int = TOP_K,
                          seed: int = SEED) -> dict:
    """Top-k provenance values each held out once."""
    return _category_split(corpus_df, "provenance", "source_held_out",
                           top_k, seed)


def build_object_held_out(corpus_df, top_k: int = TOP_K,
                          seed: int = SEED) -> dict:
    """Top-k sub_genre (object type) values each held out once."""
    return _category_split(corpus_df, "sub_genre", "object_held_out",
                           top_k, seed)


def build_all(corpus_df, seed: int = SEED) -> dict:
    """All five splits, keyed by name (= file stem)."""
    return {s["name"]: s for s in (
        build_gkf_ruler(corpus_df, seed=seed),
        build_mc_balanced(corpus_df, seed=seed),
        build_loro(corpus_df, seed=seed),
        build_source_held_out(corpus_df, seed=seed),
        build_object_held_out(corpus_df, seed=seed))}


def to_json_bytes(split: dict) -> bytes:
    """Canonical encoding: sorted keys (ids already sorted by _fold),
    2-space indent, ASCII, trailing newline — byte-identical across
    runs by construction."""
    return (json.dumps(split, sort_keys=True, indent=2,
                       ensure_ascii=True) + "\n").encode("utf-8")


def write_split(split: dict, out_dir: str | None = None) -> str:
    out_dir = out_dir or os.path.join(common.ART, "splits")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split['name']}.json")
    with open(path, "wb") as f:
        f.write(to_json_bytes(split))
    return path
