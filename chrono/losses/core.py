"""Differentiable loss library for the chrono head (SLA section 5).

WHAT. Pure tensor functions — no nn.Module state — implementing the terms
the trainer (A4) composes: Barlow-Twins redundancy reduction between two
views, soft-rank ordering losses over weak (i earlier than j) pairs, a
differentiable Spearman, anti-collapse variance, an RBF-HSIC independence
penalty, graph smoothness, and a Gaussian interval NLL.

WHY pure functions: every term must be gradcheck-able in isolation and
composable under any weighting without hidden buffers, so exactness is
testable term by term (plan P2.1-P2.6). Everything is dtype-agnostic
(tests run in double for torch.autograd.gradcheck) and CPU-safe.

Convention (SLA section 1): every score `s` is a LATENESS score (larger =
later, astronomical t). An ordered pair (i, j) always means t_i < t_j.
"""
from __future__ import annotations

import math

import torch


def _standardize(z: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-dim batch standardization with biased variance, so a dim's
    autocorrelation is exactly var/(var+eps) ~ 1 and identical views give
    an on-diagonal Barlow term of ~0."""
    mu = z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False, keepdim=True)
    return (z - mu) / torch.sqrt(var + eps)


def bt_loss(z_a: torch.Tensor, z_b: torch.Tensor, *,
            lambda_offdiag: float = 5e-3,
            eps: float = 1e-6) -> torch.Tensor:
    """Barlow Twins: cross-correlate the two views' batch-standardized
    projections; pull the diagonal to 1 (invariance) and the off-diagonal
    to 0 (redundancy reduction). z_a, z_b: [B, D]."""
    if z_a.shape != z_b.shape or z_a.dim() != 2:
        raise ValueError(f"expected matching [B, D], got "
                         f"{tuple(z_a.shape)} vs {tuple(z_b.shape)}")
    b = z_a.shape[0]
    c = _standardize(z_a, eps).T @ _standardize(z_b, eps) / b
    on = torch.diagonal(c)
    off = c - torch.diag_embed(on)
    return ((on - 1.0) ** 2).sum() + lambda_offdiag * (off ** 2).sum()


def softrank_loss(s: torch.Tensor, pairs: torch.Tensor,
                  margins: torch.Tensor, *,
                  temp: float = 1.0) -> torch.Tensor:
    """Margin ranking surrogate over weak-order pairs.

    s: [N] lateness scores; pairs: Long[P, 2] rows (i, j) with t_i < t_j;
    margins: [P] >= 0. mean softplus((s[i] - s[j] + margin) / temp): zero
    once s[j] >= s[i] + margin, and (margin-scaling property, tested) at
    s[i] == s[j] the per-pair gradient magnitude sigmoid(m/temp)/temp
    GROWS with the margin — far pairs push harder than adjacent ones.
    """
    gap = s[pairs[:, 0]] - s[pairs[:, 1]] + margins
    return torch.nn.functional.softplus(gap / temp).mean()


def _soft_rank(x: torch.Tensor, temp: float) -> torch.Tensor:
    """All-pairs sigmoid soft rank: r_i = sum_j sigmoid((x_i - x_j)/temp).
    The constant self-term (0.5) shifts every rank equally and cancels in
    any Pearson. No torchsort (SLA env: cluster-only packages banned)."""
    return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / temp).sum(1)


def soft_spearman(s: torch.Tensor, t: torch.Tensor, *,
                  temp: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman in [-1, 1]: Pearson of the all-pairs
    sigmoid soft ranks of s and t. As temp -> 0 soft ranks approach hard
    ranks, recovering scipy.stats.spearmanr (tested at temp <= 0.05)."""
    t = torch.as_tensor(t, dtype=s.dtype, device=s.device)
    ra = _soft_rank(s.reshape(-1), temp)
    rb = _soft_rank(t.reshape(-1), temp)
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = torch.sqrt((da ** 2).sum() * (db ** 2).sum() + 1e-12)
    return (da * db).sum() / denom


def variance_loss(s: torch.Tensor, *, floor: float = 1.0) -> torch.Tensor:
    """Anti-collapse hinge on the batch std (VICReg-style): penalize each
    dim's std falling below `floor`. s: [N] scores or [N, D] features;
    mean hinge over dims."""
    m = s.reshape(s.shape[0], -1)
    std = torch.sqrt(m.var(dim=0, unbiased=False) + 1e-12)
    return torch.relu(floor - std).mean()


def _sq_dists(x: torch.Tensor) -> torch.Tensor:
    d = x.unsqueeze(1) - x.unsqueeze(0)
    return (d ** 2).sum(-1)


def _median_sigma(sq: torch.Tensor) -> float:
    """Median-heuristic bandwidth from squared distances (detached: the
    bandwidth is a scale choice, not a gradient path). Falls back to 1.0
    for degenerate all-equal inputs."""
    with torch.no_grad():
        off = sq[~torch.eye(sq.shape[0], dtype=torch.bool,
                            device=sq.device)]
        med = off[off > 0].median() if (off > 0).any() else None
    return float(torch.sqrt(med)) if med is not None else 1.0


def hsic_loss(x: torch.Tensor, y: torch.Tensor, *,
              sigma_x: float | None = None,
              sigma_y: float | None = None) -> torch.Tensor:
    """Biased empirical RBF-HSIC (Gretton et al. 2005):
    tr(K H L H) / (n-1)^2 with K_ij = exp(-||x_i-x_j||^2 / (2 sigma^2)).
    ~0 iff x and y are independent; unlike Pearson it detects NONLINEAR
    dependence (tested: x vs x^2 with Pearson ~ 0). Median-heuristic
    bandwidths when sigma is None. x, y: [N] or [N, D]."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y need the same batch size")
    sqx, sqy = _sq_dists(x), _sq_dists(y)
    sx = _median_sigma(sqx) if sigma_x is None else float(sigma_x)
    sy = _median_sigma(sqy) if sigma_y is None else float(sigma_y)
    k = torch.exp(-sqx / (2.0 * sx ** 2))
    l = torch.exp(-sqy / (2.0 * sy ** 2))
    h = (torch.eye(n, dtype=x.dtype, device=x.device)
         - 1.0 / n)
    return torch.trace(k @ h @ l @ h) / (n - 1) ** 2


def graph_smoothness(s: torch.Tensor, edges: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
    """Weighted Dirichlet energy sum_e w_e (s_i - s_j)^2 over an edge
    list. edges: Long[E, 2]; weights: [E] >= 0. Small when scores vary
    smoothly along the (temporal-adjacency) graph."""
    d = s[edges[:, 0]] - s[edges[:, 1]]
    return (weights * d ** 2).sum()


def interval_nll(mu: torch.Tensor, sigma: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood of true dates t under
    predicted (mu, sigma). sigma must already be positive — the head
    applies softplus upstream (SLA section 5); no clamping here so the
    gradient through sigma stays exact."""
    z = (t - mu) / sigma
    return (0.5 * z ** 2 + torch.log(sigma)
            + 0.5 * math.log(2.0 * math.pi)).mean()
