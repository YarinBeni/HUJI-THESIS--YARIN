"""PRIVATE fallback shims for the A3 loss contract (SLA section 5).

WHAT. Minimal, correct implementations of exactly the five signatures
train_cjb.py consumes — bt_loss, softrank_loss, soft_spearman,
variance_loss, make_order_pairs — used ONLY when `chrono.losses` cannot
be imported (the library is built by a sibling agent in parallel).
train_cjb imports chrono.losses first and falls back here on
ImportError with a loud warning, so integration automatically prefers
the real library the moment it lands.

WHY duplicated at all: the trainer's overfit gate must pass standalone
on CPU (definition of done) without waiting on the sibling. Keep these
in lockstep with INTERFACES.md section 5 and nothing more; anything
fancier belongs in chrono/losses/.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch


def _standardize(z, eps):
    mu = z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False, keepdim=True)
    return (z - mu) / torch.sqrt(var + eps)


def bt_loss(z_a, z_b, *, lambda_offdiag=5e-3, eps=1e-6):
    """Barlow Twins over [B, D] views: on-diag -> 1, off-diag -> 0."""
    b = z_a.shape[0]
    c = _standardize(z_a, eps).T @ _standardize(z_b, eps) / b
    on = torch.diagonal(c)
    off = c - torch.diag_embed(on)
    return ((on - 1.0) ** 2).sum() + lambda_offdiag * (off ** 2).sum()


def softrank_loss(s, pairs, margins, *, temp=1.0):
    """mean softplus((s[i] - s[j] + margin)/temp), (i, j): t_i < t_j."""
    gap = s[pairs[:, 0]] - s[pairs[:, 1]] + margins
    return torch.nn.functional.softplus(gap / temp).mean()


def _soft_rank(x, temp):
    return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / temp).sum(1)


def soft_spearman(s, t, *, temp=1.0):
    """Pearson of all-pairs sigmoid soft ranks of s and t, in [-1, 1]."""
    t = torch.as_tensor(t, dtype=s.dtype, device=s.device)
    ra, rb = _soft_rank(s.reshape(-1), temp), _soft_rank(t.reshape(-1),
                                                         temp)
    da, db = ra - ra.mean(), rb - rb.mean()
    return (da * db).sum() / torch.sqrt(
        (da ** 2).sum() * (db ** 2).sum() + 1e-12)


def variance_loss(s, *, floor=1.0):
    """Hinge on std(s): zero once the score spread clears `floor`."""
    return torch.relu(floor - s.reshape(-1).std(unbiased=False))


def make_order_pairs(doc_df, ruler_table, *, per_ruler_pair=21, seed):
    """(pairs Long[P,2], margins float32[P], meta_df) from disjoint
    reign-proxy intervals; positional indices into doc_df; margin = the
    interval-edge gap standardized by std(t); ZERO pairs when the
    intervals overlap or touch."""
    rng = np.random.default_rng(seed)
    t = doc_df["t"].to_numpy(dtype=float)
    t_std = float(np.std(t)) or 1.0
    iv = {r.ruler: (float(r.t_min), float(r.t_max))
          for r in ruler_table.itertuples()}
    rulers = doc_df["ruler"].to_numpy()
    pos = {r: np.flatnonzero(rulers == r)
           for r in sorted(set(rulers) & set(iv))}
    ij, mg, rows = [], [], []
    for ra, rb in combinations(sorted(pos), 2):
        (a0, a1), (b0, b1) = iv[ra], iv[rb]
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
        ij.append(np.stack([i, j], axis=1))
        mg.append(np.full(k, gap / t_std))
        rows.append(pd.DataFrame({
            "ruler_i": re_, "ruler_j": rl,
            "doc_i": doc_df["doc_id"].to_numpy()[i],
            "doc_j": doc_df["doc_id"].to_numpy()[j],
            "gap": gap, "margin": gap / t_std}))
    if not ij:
        return (torch.zeros((0, 2), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32), pd.DataFrame())
    return (torch.as_tensor(np.concatenate(ij), dtype=torch.long),
            torch.as_tensor(np.concatenate(mg), dtype=torch.float32),
            pd.concat(rows, ignore_index=True))
