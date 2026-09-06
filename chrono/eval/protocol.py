"""Evaluation protocol — how a lateness score becomes a rho (SLA §7).

WHAT. Three read-outs over pre-frozen splits in the SLA §3 JSON shape
({"name", "kind", "seed", "folds": [{"train": [...], "test": [...]}]}):

  * mc_balanced_rho — one Spearman rho per draw. Each fold of an
    mc_balanced split IS one balanced draw, and rho is computed over
    that draw's test docs only, scores vs the corpus t column.
  * gkf_rho — one rho per held-out fold. GroupKFold scores come from a
    different model per fold (fitted on that fold's train side), so the
    caller hands one score Series per fold index.
  * placebo_rho — mc_balanced_rho with t shuffled independently within
    each draw via default_rng(seed); an honest pipeline straddles 0.

WHY centralized. The recurring failure mode of this project is a scorer
that also picks its own evaluation frame. Here a scorer hands over
per-doc lateness scores (larger = later) and nothing else: t comes from
the corpus (astronomical years, larger = later, SLA §1) and is NEVER
re-derived; test lists come from the frozen split JSONs; a doc missing
from scores or corpus is a hard, named error — never a silent drop.

Pure numpy/scipy/pandas; no torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["mc_balanced_rho", "gkf_rho", "placebo_rho"]


def _folds(split: dict) -> list:
    if not isinstance(split, dict) or "folds" not in split:
        got = (sorted(split) if isinstance(split, dict)
               else type(split).__name__)
        raise ValueError("split must be a dict with a 'folds' key "
                         f"(SLA §3 JSON shape); got {got}")
    folds = split["folds"]
    if not folds:
        raise ValueError(f"split {split.get('name', '?')!r} has no folds")
    return folds


def _t_by_doc(corpus_df: pd.DataFrame) -> pd.Series:
    for col in ("doc_id", "t"):
        if col not in corpus_df.columns:
            raise ValueError(f"corpus_df lacks required column {col!r}")
    t = pd.Series(corpus_df["t"].to_numpy(dtype=float),
                  index=pd.Index(corpus_df["doc_id"], name="doc_id"))
    if t.index.has_duplicates:
        dup = t.index[t.index.duplicated()][:5].tolist()
        raise ValueError(f"corpus_df has duplicate doc_id(s): {dup}")
    return t


def _check_scores(scores: pd.Series, what: str = "scores") -> pd.Series:
    if not isinstance(scores, pd.Series):
        raise TypeError(f"{what} must be a pd.Series indexed by doc_id, "
                        f"got {type(scores).__name__}")
    if scores.index.has_duplicates:
        dup = scores.index[scores.index.duplicated()][:5].tolist()
        raise ValueError(f"{what} index has duplicate doc_id(s): {dup}")
    return scores


def _gather(scores: pd.Series, t_by_doc: pd.Series, test_ids,
            where: str) -> tuple:
    """Align (s, t) over one fold's test docs; any missing doc raises."""
    idx = pd.Index(test_ids)
    miss = idx.difference(t_by_doc.index)
    if len(miss):
        raise KeyError(f"{len(miss)} test doc(s) of {where} missing from "
                       f"corpus_df: {miss[:5].tolist()}")
    miss = idx.difference(scores.index)
    if len(miss):
        raise KeyError(f"{len(miss)} test doc(s) of {where} missing from "
                       f"scores: {miss[:5].tolist()}")
    return (scores.loc[idx].to_numpy(dtype=float),
            t_by_doc.loc[idx].to_numpy(dtype=float))


def _spearman(s: np.ndarray, t: np.ndarray) -> float:
    """rho, or nan when undefined (n < 2, or a constant input)."""
    if s.size < 2 or np.all(s == s[0]) or np.all(t == t[0]):
        return float("nan")
    return float(stats.spearmanr(s, t).statistic)


def _per_draw_rho(scores: pd.Series, corpus_df: pd.DataFrame, split: dict,
                  perm_rng=None) -> np.ndarray:
    scores = _check_scores(scores)
    t_by_doc = _t_by_doc(corpus_df)
    name = split.get("name", "?") if isinstance(split, dict) else "?"
    out = []
    for k, fold in enumerate(_folds(split)):
        s, t = _gather(scores, t_by_doc, fold["test"],
                       f"split {name!r} fold {k}")
        if perm_rng is not None:
            t = perm_rng.permutation(t)
        out.append(_spearman(s, t))
    return np.asarray(out, dtype=float)


def mc_balanced_rho(scores: pd.Series, corpus_df: pd.DataFrame,
                    split: dict) -> np.ndarray:
    """Spearman rho per draw of an mc-style split, scores vs corpus t.

    scores: pd.Series of lateness scores indexed by doc_id (one frozen
    scoring of the corpus — mc draws never refit). Returns float array
    of len(split["folds"]); draw k uses folds[k]["test"] docs only.
    """
    return _per_draw_rho(scores, corpus_df, split)


def gkf_rho(scores_by_fold: dict, corpus_df: pd.DataFrame,
            split: dict) -> np.ndarray:
    """Spearman rho per fold, each fold scored by its own model.

    scores_by_fold: {fold_index: pd.Series indexed by doc_id}, one entry
    per fold of the split (indices 0..K-1 in folds order); every fold's
    Series must cover that fold's test docs. Returns float array [K].
    """
    folds = _folds(split)
    missing = [k for k in range(len(folds)) if k not in scores_by_fold]
    if missing:
        raise KeyError(f"scores_by_fold missing fold index(es) {missing} "
                       f"(split has {len(folds)} folds)")
    t_by_doc = _t_by_doc(corpus_df)
    name = split.get("name", "?")
    out = []
    for k, fold in enumerate(folds):
        scores = _check_scores(scores_by_fold[k], f"scores_by_fold[{k}]")
        s, t = _gather(scores, t_by_doc, fold["test"],
                       f"split {name!r} fold {k}")
        out.append(_spearman(s, t))
    return np.asarray(out, dtype=float)


def placebo_rho(scores: pd.Series, corpus_df: pd.DataFrame, split: dict,
                seed: int) -> np.ndarray:
    """mc_balanced_rho under the null: t permuted within each draw.

    One default_rng(seed) drives all draws in order, so the whole curve
    is reproducible from (split, seed). A sound pipeline must straddle 0
    here; a placebo that doesn't is a leak detector firing.
    """
    return _per_draw_rho(scores, corpus_df, split,
                         perm_rng=np.random.default_rng(seed))

def pooled_rho(scores: pd.Series, corpus_df: pd.DataFrame,
               split: dict) -> float:
    """ONE Spearman over the concatenation of every fold's test docs.

    REVIEW FIX (wave B1). Per-fold rho is undefined wherever a fold holds
    a single ruler: 39 of our 40 rulers carry exactly ONE distinct year,
    so leave-one-ruler-out folds have constant t by construction and
    gkf folds 0/1 (the two mega-rulers) do too. Averaging the surviving
    folds silently changes the measurand — a LORO cell computed that way
    is a WITHIN-reign number wearing an unseen-ruler label.

    Pooling is the honest read-out for those split kinds: the claim
    "unseen rulers land in the right chronological place" is exactly a
    correlation across the pooled held-out docs. Scores must already be
    out-of-fold (each doc scored by a model that did not train on it);
    with a single frozen scoring this is a transductive number and the
    caller is responsible for saying so.
    """
    t_by_doc = _t_by_doc(corpus_df)
    scores = _check_scores(scores)
    ids = [d for f in _folds(split) for d in f.get("test", [])]
    if not ids:
        return float("nan")
    s, t = _gather(scores, t_by_doc, ids, "pooled")
    return _spearman(s, t)


def block_placebo_rho(scores: pd.Series, corpus_df: pd.DataFrame,
                      split: dict, seed: int) -> np.ndarray:
    """The placebo at the RIGHT exchangeable unit: permute the ruler->t
    assignment, docs inherit their ruler's permuted year.

    REVIEW FIX (wave B1). placebo_rho shuffles t per DOC, but t is
    block-constant within ruler (39/40 rulers = one year), so the
    doc-level null is far too narrow: measured on the real artifacts its
    95% band is about [-0.16, +0.18] against [-0.51, +0.47] for this
    block null. Doc-level stays as a LEAK DETECTOR; significance claims
    must be made against this one, because the effective sample size is
    the number of rulers in a draw (8), not its 168 documents.
    """
    t_by_doc = _t_by_doc(corpus_df)
    ruler = corpus_df.set_index("doc_id")["ruler"]
    rng = np.random.default_rng(seed)
    scores = _check_scores(scores)
    out = []
    for f in _folds(split):
        ids = list(f.get("test", []))
        if not ids:
            out.append(float("nan"))
            continue
        rl = ruler.loc[ids]
        uniq = list(dict.fromkeys(rl.tolist()))
        true_t = {r: float(t_by_doc.loc[rl.index[rl == r]].iloc[0])
                  for r in uniq}
        perm = rng.permutation(len(uniq))
        mapped = {r: true_t[uniq[perm[i]]] for i, r in enumerate(uniq)}
        s = scores.loc[ids].to_numpy(dtype=float)
        t = np.array([mapped[r] for r in rl], dtype=float)
        out.append(_spearman(s, t))
    return np.asarray(out, dtype=float)
