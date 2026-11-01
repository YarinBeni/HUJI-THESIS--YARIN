"""AdapterHead + EMA twin — the trainable surface of Chrono-Barlow
(plan P3.1/P4.1; SLA section 6).

WHAT. AdapterHead sits on top of FROZEN features (cached encoder
embeddings, or tfidf on the local smoke path) and exposes the three
outputs the losses consume: h = mlp(x) the adapted representation,
s = axis(h) the scalar LATENESS score (larger = later, SLA section 1),
p = proj(h) the projector output the Barlow-Twins term correlates.
EmaTwin wraps a frozen exponential-moving-average copy of a head as the
JEPA target branch: forward is stop-grad, and .update() applies
xi <- m*xi + (1 - m)*theta after every optimizer step.

WHY a separate scalar axis: the thesis question is whether ONE direction
carries composition-time order; keeping s a rank-1 readout of h makes
the later erasure/leakage probes (P3.5) interpretable, while the wider
projector p absorbs the invariance pressure so BT cannot collapse the
axis itself.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class AdapterHead(nn.Module):
    """MLP adapter over frozen d_in features -> (h, s, p)."""

    def __init__(self, d_in: int, d_hidden: int = 512,
                 d_proj: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.axis = nn.Linear(d_hidden, 1)
        self.proj = nn.Linear(d_hidden, d_proj)

    def forward(self, x: torch.Tensor):
        h = self.mlp(x)
        s = self.axis(h).squeeze(-1)
        p = self.proj(h)
        return h, s, p


class EmaTwin(nn.Module):
    """Stop-grad EMA copy of `head` (target branch, P4.1).

    The online head is held by reference OUTSIDE the module tree (a
    tuple defeats nn.Module attribute registration) so that
    twin.parameters() is the target only — an optimizer built over it
    can never touch the online weights, and state_dict stays clean.
    """

    def __init__(self, head: nn.Module, momentum: float = 0.996):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1]: {momentum}")
        self.momentum = float(momentum)
        self._online = (head,)
        self.target = copy.deepcopy(head)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            return self.target(x)

    @torch.no_grad()
    def update(self):
        """xi <- m*xi + (1 - m)*theta over params; buffers copied."""
        m, online = self.momentum, self._online[0]
        for pt, po in zip(self.target.parameters(),
                          online.parameters(), strict=True):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
        for bt, bo in zip(self.target.buffers(),
                          online.buffers(), strict=True):
            bt.copy_(bo)
