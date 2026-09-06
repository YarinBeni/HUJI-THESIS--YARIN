"""Ordered-pair generator from reign-proxy weak labels (P2.7).

WHAT. Turns the per-ruler interval table [t_min, t_max] (a REIGN PROXY —
each ruler's fragment years, SLA section 3) into training pairs for
softrank_loss: (i, j) positional indices into doc_df with t_i < t_j,
plus a margin per pair. A cross-ruler pair is emitted ONLY when the two
intervals are strictly disjoint, because only then does EVERY fragment
of the earlier ruler provably predate every fragment of the later one —
overlapping or touching intervals yield ZERO pairs (tested). The margin
is the gap between the interval edges, standardized by std(t) over
doc_df, so far-apart reigns demand a wider score separation.

WHY quota + weights: fragment counts per ruler are wildly skewed, and
pairing SQUARES the skew (see v_1/src/phase2/pairs/pairs_data.py, whose
draw_pairs this mirrors). The balancing unit is the RULER-PAIR: each
contributes min(per_ruler_pair, n_i, n_j) pairs, docs sampled without
replacement within the pair, and meta carries weight = 1/k so every
ruler-pair can carry equal total loss downstream. Deterministic by seed.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch

META_COLS = ["ruler_i", "ruler_j", "doc_i", "doc_j", "pos_i", "pos_j",
             "t_i", "t_j", "gap", "margin", "weight"]


MARGIN_MAX = 2.0        # see _margin: the achievable score scale


def _margin(gap, t_std):
    """Squash the reign gap into a margin the score scale can satisfy.

    REVIEW FIX (wave B1). margin = gap/std(t) ran to 9.99 while
    variance_loss floors std(s) at 1.0, so 36% of pairs sat permanently
    in softplus's saturated regime: constant gradient regardless of the
    actual ordering error, i.e. force allocated by reign distance rather
    than by violation. tanh keeps the ordering of margins (far pairs
    still ask for more separation) but bounds them inside the range a
    unit-variance axis can actually deliver.
    """
    return MARGIN_MAX * np.tanh(np.asarray(gap, dtype=float) / t_std)


def make_order_pairs(doc_df: pd.DataFrame, ruler_table: pd.DataFrame, *,
                     t_std: float = None,
                     per_ruler_pair: int = 21, seed: int
                     ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Build (pairs, margins, meta_df) from disjoint reign intervals.

    doc_df: corpus frame (doc_id, ruler, t, ...); pair indices are
    POSITIONAL row numbers of doc_df, aligning with a score vector s of
    len(doc_df). ruler_table: ruler, t_min, t_max (astronomical). Rulers
    present in only one of the two frames are skipped. Returns
    pairs Long[P, 2] with row (i, j) meaning i earlier, margins
    float32[P], and meta_df (META_COLS) with one row per pair.
    """
    if per_ruler_pair < 1:
        raise ValueError("per_ruler_pair must be >= 1")
    rng = np.random.default_rng(seed)
    t_all = doc_df["t"].to_numpy(dtype=float)
    # REVIEW FIX: t_std must be a CORPUS constant, not a fold statistic,
    # or the same reign gap asks for different separations in different
    # folds (measured fold std(t): 101-127) and margins stop being
    # comparable across the E-MIN grid.
    t_std = float(t_std if t_std is not None else np.std(t_all))
    if not t_std > 0:
        raise ValueError("std(t) over doc_df must be > 0 for margins")

    interval = {r.ruler: (float(r.t_min), float(r.t_max))
                for r in ruler_table.itertuples()}
    rulers = doc_df["ruler"].to_numpy()
    pos = {r: np.flatnonzero(rulers == r)
           for r in sorted(set(rulers) & set(interval))}

    ij, rows = [], []
    for ra, rb in combinations(sorted(pos), 2):
        (a0, a1), (b0, b1) = interval[ra], interval[rb]
        # earlier ruler first; strictly disjoint means edge < edge —
        # touching intervals (a1 == b0) still overlap at a point: skip
        if a1 < b0:
            re_, rl, gap = ra, rb, b0 - a1
        elif b1 < a0:
            re_, rl, gap = rb, ra, a0 - b1
        else:
            continue
        pe, pl = pos[re_], pos[rl]
        k = min(per_ruler_pair, len(pe), len(pl))
        i = pe[rng.choice(len(pe), size=k, replace=False)]
        j = pl[rng.choice(len(pl), size=k, replace=False)]
        margin = float(_margin(gap, t_std))
        ij.append(np.stack([i, j], axis=1))
        rows.append(pd.DataFrame({
            "ruler_i": re_, "ruler_j": rl,
            "doc_i": doc_df["doc_id"].to_numpy()[i],
            "doc_j": doc_df["doc_id"].to_numpy()[j],
            "pos_i": i, "pos_j": j,
            "t_i": t_all[i], "t_j": t_all[j],
            "gap": gap, "margin": margin, "weight": 1.0 / k}))

    if not ij:
        return (torch.zeros((0, 2), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                pd.DataFrame(columns=META_COLS))
    meta = pd.concat(rows, ignore_index=True)[META_COLS]
    pairs = torch.as_tensor(np.concatenate(ij), dtype=torch.long)
    margins = torch.tensor(meta["margin"].to_numpy(),
                           dtype=torch.float32)
    return pairs, margins, meta
