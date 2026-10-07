"""Robustness battery — every condition x every split, one table (P3.4).

WHAT. battery() pivots a long scores table (doc_id, condition, s) into
one row per condition x split carrying the per-fold Spearman aggregate
{rho_mean, rho_sd, n}. Conditions are scoring variants of the SAME docs
(orig, mask_ruler, strip_formula, crop16, ...); splits are the frozen
SLA §3 dicts (mc_balanced, gkf_ruler, loro, source/object held-out).
Every cell goes through protocol.mc_balanced_rho — rho over each fold's
test docs vs corpus t — so a battery cell and a headline number cannot
disagree by construction. This is legitimate for ANY split kind here
because the battery evaluates one frozen scoring per condition; per-fold
refit protocols report through gkf_rho instead, upstream of this table.

WHY. The thesis claim is not "some rho is high" but "the rho survives
having its crutches kicked away": still there when ruler names are
masked, when formulae are stripped, when whole rulers / find-spots /
object types are held out. That claim is only auditable as a single
complete grid — which is exactly what this module emits, with degenerate
folds (rho undefined: <2 test docs or constant input) dropped from the
aggregate and n reporting how many folds actually contributed, so a
hollow cell is visible instead of silently averaged over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chrono.eval.protocol import mc_balanced_rho, pooled_rho

__all__ = ["battery", "BATTERY_COLS"]

BATTERY_COLS = ["condition", "split", "readout", "rho_mean",
                "rho_sd", "n"]
MC_SPLITS = frozenset({"mc_balanced"})


def battery(scores_df: pd.DataFrame, corpus_df: pd.DataFrame,
            splits: dict) -> pd.DataFrame:
    """Condition x split grid of per-fold rho aggregates.

    scores_df: long table with columns {doc_id, condition, s}; one row
    per (condition, doc_id), s a lateness score (larger = later).
    splits: {split_name: split_dict} in the SLA §3 JSON shape.
    Returns a DataFrame with exactly BATTERY_COLS; rows ordered by
    condition first-appearance then splits insertion order. rho_sd uses
    ddof=1 (nan when n < 2); n counts folds with a defined rho.

    READ-OUT POLICY (review fix, wave B1). Per-fold rho is only defined
    where a fold spans several years. With 39 of 40 rulers carrying one
    distinct year, that holds for the mc draws (8 rulers each) but NOT
    for leave-one-ruler-out, gkf (its two mega-ruler folds) or the
    held-out-category splits. Those are therefore POOLED: one rho over
    the concatenated held-out docs (`pooled_rho`), reported with
    readout='pooled', n = docs. Averaging per-fold rho there would
    silently answer a within-reign question instead. A split name is
    routed by MC_SPLITS membership, so a new mc-style split must be
    registered there.
    """
    need = {"doc_id", "condition", "s"}
    missing = need - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df missing column(s) {sorted(missing)}")
    dup = scores_df.duplicated(["condition", "doc_id"])
    if dup.any():
        pairs = (scores_df.loc[dup, ["condition", "doc_id"]]
                 .head(5).to_records(index=False).tolist())
        raise ValueError("scores_df has duplicate (condition, doc_id) "
                         f"rows, e.g. {pairs}")
    if not splits:
        raise ValueError("splits dict is empty")

    conditions = list(dict.fromkeys(scores_df["condition"]))
    rows = []
    for cond in conditions:
        sub = scores_df[scores_df["condition"] == cond]
        s = pd.Series(sub["s"].to_numpy(dtype=float),
                      index=pd.Index(sub["doc_id"], name="doc_id"))
        for split_name, split in splits.items():
            if split_name in MC_SPLITS:
                rhos = mc_balanced_rho(s, corpus_df, split)
                fin = rhos[np.isfinite(rhos)]
                n = int(fin.size)
                rows.append(dict(
                    condition=cond, split=split_name, readout="per_draw",
                    rho_mean=float(fin.mean()) if n else float("nan"),
                    rho_sd=float(fin.std(ddof=1)) if n > 1
                    else float("nan"),
                    n=n))
            else:
                rho = pooled_rho(s, corpus_df, split)
                n = sum(len(f.get("test", [])) for f in split["folds"])
                rows.append(dict(
                    condition=cond, split=split_name, readout="pooled",
                    rho_mean=float(rho), rho_sd=float("nan"), n=int(n)))
    return pd.DataFrame(rows, columns=BATTERY_COLS)
