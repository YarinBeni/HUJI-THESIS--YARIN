"""Pooling sites for the stress-test probes.

Three sites (user-locked):
  * mean       — attention-masked mean over all tokens (whole text). tier0 + maximal.
  * king_last  — hidden state of the LAST token of the located king-name span. tier0 only.
  * king_mean  — mean of the king-name span's tokens.                       tier0 only.

These operate on a single example's hidden states (seq_len, hidden) as numpy or
torch; the extractor (J4) calls them per fragment after locating the name span
with ``king_token.name_token_span``. Whole-sentence last-token pooling is
intentionally NOT provided (dropped per Yarin).
"""
from __future__ import annotations

import numpy as np


def _to_np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)


def pool_mean(hidden, attention_mask=None):
    """hidden: (seq_len, hidden). attention_mask: (seq_len,) of 0/1 or None."""
    h = _to_np(hidden)
    if attention_mask is None:
        return h.mean(axis=0)
    m = _to_np(attention_mask).reshape(-1, 1)
    s = (h * m).sum(axis=0)
    c = max(float(m.sum()), 1.0)
    return s / c


def pool_king_last(hidden, span):
    """span: (tok_start, tok_end) inclusive, from king_token.name_token_span."""
    if span is None:
        return None
    return _to_np(hidden)[span[1]]


def pool_king_mean(hidden, span):
    if span is None:
        return None
    h = _to_np(hidden)
    return h[span[0]: span[1] + 1].mean(axis=0)


POOLERS = {"mean": pool_mean, "king_last": pool_king_last, "king_mean": pool_king_mean}
