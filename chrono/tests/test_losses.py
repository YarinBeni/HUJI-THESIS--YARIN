"""A3 loss library tests (plan P2.1-P2.7, SLA section 5): exactness of
every term — gradcheck in double, behavior on synthetic data with known
answers, and the weak-label pair generator's disjointness guarantee."""
import numpy as np
from chrono.losses import calibrate, core
import pandas as pd
import pytest
import torch
from scipy import stats
from torch.autograd import gradcheck

from chrono.losses import (MonotoneCalibrator, bt_loss, graph_smoothness,
                           hsic_loss, interval_nll, make_order_pairs,
                           soft_spearman, softrank_loss, variance_loss)


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------- P2.1
def test_bt_identical_views_offdiag_only():
    z = _randn(64, 16, seed=5)
    on_only = bt_loss(z, z, lambda_offdiag=0.0)
    assert float(on_only) < 1e-6          # diag of the cross-corr is ~1
    full = bt_loss(z, z, lambda_offdiag=5e-3)
    off_term = float(full - on_only)
    assert off_term > 0                   # random dims correlate a bit
    assert float(full) == pytest.approx(off_term, abs=1e-6)


def test_bt_decreases_for_aligned_views():
    z = _randn(64, 8, seed=6)
    noisy = z + 2.0 * _randn(64, 8, seed=7)
    assert float(bt_loss(z, z)) < float(bt_loss(z, noisy))


# ----------------------------------------------------- gradchecks (all)
def test_gradcheck_bt():
    za = _randn(5, 3, seed=1).requires_grad_()
    zb = _randn(5, 3, seed=2).requires_grad_()
    assert gradcheck(lambda a, b: bt_loss(a, b), (za, zb))


def test_gradcheck_softrank():
    s = _randn(6, seed=3).requires_grad_()
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 4]])
    margins = torch.tensor([0.1, 0.5, 1.0, 0.2], dtype=torch.float64)
    assert gradcheck(
        lambda v: softrank_loss(v, pairs, margins, temp=0.7), (s,))


def test_gradcheck_soft_spearman():
    s = _randn(6, seed=4).requires_grad_()
    t = _randn(6, seed=5)
    assert gradcheck(lambda v: soft_spearman(v, t, temp=0.5), (s,))


def test_gradcheck_variance():
    # scaled down so the hinge is ACTIVE (std < floor) and smooth there
    v = (0.1 * _randn(7, seed=6)).requires_grad_()
    assert gradcheck(lambda u: variance_loss(u, floor=1.0), (v,))


def test_gradcheck_hsic():
    # fixed sigmas: the median-heuristic bandwidth is detached by design
    # (a scale choice, not a gradient path), so finite differences would
    # disagree with the analytical gradient through a moving median.
    x = _randn(6, seed=7).requires_grad_()
    y = _randn(6, seed=8).requires_grad_()
    assert gradcheck(
        lambda a, b: hsic_loss(a, b, sigma_x=1.0, sigma_y=1.3), (x, y))


def test_gradcheck_graph_smoothness():
    s = _randn(6, seed=9).requires_grad_()
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    g = torch.Generator().manual_seed(10)
    w = torch.rand(5, generator=g, dtype=torch.float64)
    assert gradcheck(lambda u: graph_smoothness(u, edges, w), (s,))


def test_gradcheck_interval_nll():
    mu = _randn(6, seed=11).requires_grad_()
    g = torch.Generator().manual_seed(12)
    sg = (torch.rand(6, generator=g, dtype=torch.float64)
          + 0.5).requires_grad_()
    t = _randn(6, seed=13)
    assert gradcheck(lambda m, q: interval_nll(m, q, t), (mu, sg))


# ---------------------------------------------------------------- P2.2
def test_softrank_training_recovers_order():
    """200 Adam steps on all ordered pairs of 50 docs must drive the
    true (scipy) Spearman of scores vs t to >= .99."""
    t = _randn(50, seed=0)
    tn = t.numpy()
    std = tn.std()
    ii, jj = np.nonzero(tn[:, None] < tn[None, :])
    pairs = torch.tensor(np.stack([ii, jj], 1), dtype=torch.long)
    margins = torch.tensor((tn[jj] - tn[ii]) / std, dtype=torch.float64)
    s = (0.01 * _randn(50, seed=1)).requires_grad_()
    opt = torch.optim.Adam([s], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        softrank_loss(s, pairs, margins, temp=1.0).backward()
        opt.step()
    rho = stats.spearmanr(s.detach().numpy(), tn).statistic
    assert rho >= 0.99


def test_softrank_margin_scaling():
    """At equal scores, a pair with a larger margin must push harder:
    grad magnitude sigmoid(m/temp)/temp grows with m."""
    def grad_norm(margin):
        s = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        pairs = torch.tensor([[0, 1]])
        m = torch.tensor([margin], dtype=torch.float64)
        softrank_loss(s, pairs, m, temp=1.0).backward()
        return float(s.grad.norm())
    near, far = grad_norm(0.1), grad_norm(3.0)
    assert far > near * 1.5


def test_soft_spearman_matches_scipy_at_low_temp():
    """n=200 gaussian data across a spread of true correlations: the
    soft value tracks scipy.spearmanr with corr >= .99 at temp <= .05."""
    soft_v, sci_v = [], []
    for k, rho in enumerate([-0.9, -0.5, -0.2, 0.0, 0.3, 0.6, 0.9]):
        for rep in range(3):
            r = np.random.default_rng(97 * k + rep)
            t = r.normal(size=200)
            s = rho * t + np.sqrt(1 - rho ** 2) * r.normal(size=200)
            soft_v.append(float(soft_spearman(
                torch.tensor(s), torch.tensor(t), temp=0.05)))
            sci_v.append(stats.spearmanr(s, t).statistic)
    soft_v, sci_v = np.array(soft_v), np.array(sci_v)
    assert np.corrcoef(soft_v, sci_v)[0, 1] >= 0.99
    assert np.abs(soft_v - sci_v).max() < 0.02
    assert np.abs(soft_v).max() <= 1.0 + 1e-9


# ---------------------------------------------------------------- P2.3
def test_variance_loss_prevents_collapse():
    """An attractive-only objective (minimize var) collapses scores to a
    point; adding the hinge keeps std pinned near the floor."""
    def run(with_var):
        s = _randn(64, seed=3).requires_grad_()
        opt = torch.optim.Adam([s], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            loss = 0.5 * s.var(unbiased=False)
            if with_var:
                loss = loss + variance_loss(s, floor=1.0)
            loss.backward()
            opt.step()
        return float(s.detach().std(unbiased=False))
    assert run(with_var=False) < 0.05
    assert run(with_var=True) > 0.5


# ---------------------------------------------------------------- P2.4
def test_hsic_detects_nonlinear_dependence():
    r = np.random.default_rng(2)          # seed with |pearson| < .05
    x = torch.tensor(r.normal(size=400))
    y_ind = torch.tensor(r.normal(size=400))
    assert abs(np.corrcoef(x.numpy(), (x ** 2).numpy())[0, 1]) < 0.1
    h_dep = float(hsic_loss(x, x ** 2))
    h_ind = float(hsic_loss(x, y_ind))
    assert h_dep > 10 * h_ind             # nonlinear dep. >> independent
    assert h_ind < 0.005


# ---------------------------------------------------------------- P2.6
def test_graph_smoothness_prefers_ordered_scores():
    """Chain graph 0-1-...-9: monotone scores are smoother than shuffled
    ones, and energy shrinks as scores approach the chain ordering."""
    edges = torch.tensor([[i, i + 1] for i in range(9)])
    w = torch.ones(9, dtype=torch.float64)
    mono = torch.linspace(0, 1, 10, dtype=torch.float64)
    r = np.random.default_rng(0)
    shuf = mono[torch.tensor(r.permutation(10))]
    assert float(graph_smoothness(mono, edges, w)) \
        < float(graph_smoothness(shuf, edges, w))
    partway = 0.5 * mono + 0.5 * shuf
    assert float(graph_smoothness(mono, edges, w)) \
        < float(graph_smoothness(partway, edges, w)) \
        < float(graph_smoothness(shuf, edges, w))


# ---------------------------------------------------------------- P2.5
def test_calibrator_block_conformal_beats_doc_split():
    """REVIEW FIX (wave B1) + mutation guard.

    Rebuilds the real data's structure — t is block-constant within
    ruler — and asserts the group split is what makes leave-a-ruler-out
    coverage honest. Deleting the split, or banking per-document
    residuals under a group split, both fail this test (the old
    iid-style test passed under either mutation).
    """
    rng = np.random.default_rng(3)
    s, t, g = [], [], []
    for k in range(40):                       # 40 single-year rulers
        yr = -1100.0 + 20.0 * k
        # the real pathology: documents of one ruler are near-copies
        # (tiny within-ruler spread) while the ERROR lives at ruler level
        # (a per-ruler offset). Doc-split residuals then measure only the
        # copy noise and promise intervals far too tight for a new ruler.
        offset = rng.normal(0, 2.5)
        for _ in range(rng.integers(8, 30)):
            s.append(yr / 100.0 + offset + rng.normal(0, 0.05))
            t.append(yr)
            g.append(f"R{k}")
    s = np.array(s); t = np.array(t); g = np.array(g)

    doc = calibrate.MonotoneCalibrator(seed=0).fit(s, t)
    blk = calibrate.MonotoneCalibrator(seed=0).fit(s, t, groups=g)

    # effective n is BLOCKS under the group split, documents otherwise
    assert blk.n_effective < 0.2 * doc.n_effective

    cov_doc, _ = doc.coverage_by_group(s, t, g, 0.8)
    cov_blk, per = blk.coverage_by_group(s, t, g, 0.8)
    assert cov_blk >= 0.75                    # honest at the block level
    # strictly better, and by more than sampling jitter — this is the
    # assertion that dies if the group split is deleted (both calibrators
    # then bank the same doc-level residuals and the numbers coincide).
    # On the REAL corpus the same comparison is .88 vs .54.
    assert cov_blk > cov_doc + 0.03
    assert sum(v == 0.0 for v in per.values()) <= 4

def test_calibrator_coverage_and_monotonicity():
    r = np.random.default_rng(11)

    def make(n):
        s = r.normal(size=n)
        return s, 2.0 * s + 0.3 * s ** 3 + r.normal(size=n)

    s_tr, t_tr = make(3000)
    cal = MonotoneCalibrator(seed=0).fit(s_tr, t_tr)
    grid = np.linspace(-4, 4, 500)
    assert np.all(np.diff(cal.predict(grid)) >= 0)   # monotone map
    s_te, t_te = make(4000)
    for cov in (0.7, 0.8, 0.9):
        lo, hi = cal.predict_interval(s_te, coverage=cov)
        emp = np.mean((t_te >= lo) & (t_te <= hi))
        assert abs(emp - cov) <= 0.03, (cov, emp)
        assert np.all(lo <= hi)


# ---------------------------------------------------------------- P2.7
def _disjoint_pairs_from_table(table):
    out = set()
    iv = {r.ruler: (r.t_min, r.t_max) for r in table.itertuples()}
    for a in iv:
        for b in iv:
            if a < b:
                (a0, a1), (b0, b1) = iv[a], iv[b]
                if a1 < b0:
                    out.add((a, b))
                elif b1 < a0:
                    out.add((b, a))
    return out


def test_make_order_pairs_quota_and_order(toy_corpus, toy_ruler_table):
    pairs, margins, meta = make_order_pairs(
        toy_corpus, toy_ruler_table, per_ruler_pair=21, seed=1)
    t = toy_corpus["t"].to_numpy()
    i, j = pairs[:, 0].numpy(), pairs[:, 1].numpy()
    assert (t[i] < t[j]).all()            # every pair strictly ordered
    counts = meta.groupby(["ruler_i", "ruler_j"]).size()
    assert (counts == 21).all()           # quota: min(21, 24, 24)
    # emitted ruler-pairs are EXACTLY the table-disjoint ones
    emitted = set(map(tuple, meta[["ruler_i", "ruler_j"]]
                      .drop_duplicates().to_numpy()))
    assert emitted == _disjoint_pairs_from_table(toy_ruler_table)
    # no doc reused within a ruler-pair
    for _, g in meta.groupby(["ruler_i", "ruler_j"]):
        assert g["doc_i"].is_unique and g["doc_j"].is_unique
    # margin = interval gap / std(t), constant within a ruler-pair
    assert torch.all(margins > 0)
    # REVIEW FIX (wave B1): margins are tanh-squashed into the range a
    # unit-variance axis can satisfy, not raw gap/std(t)
    from chrono.losses.pairs import MARGIN_MAX, _margin
    assert np.allclose(meta["margin"], _margin(meta["gap"], np.std(t)))
    assert (meta["margin"] <= MARGIN_MAX).all()
    # order-preserving: bigger reign gap still asks for more separation
    o = np.argsort(meta["gap"].to_numpy())
    assert np.all(np.diff(meta["margin"].to_numpy()[o]) >= -1e-12)
    # per-ruler-pair weights sum to 1
    assert np.allclose(meta.groupby(["ruler_i", "ruler_j"])["weight"]
                       .sum(), 1.0)


def test_make_order_pairs_zero_for_overlap(toy_corpus, toy_ruler_table):
    table = toy_ruler_table.copy()
    # stretch Esarhaddon to overlap Ashurbanipal's interval
    table.loc[table.ruler == "Esarhaddon", "t_max"] = \
        float(table.loc[table.ruler == "Ashurbanipal", "t_min"].iloc[0]
              + 1.0)
    pairs, margins, meta = make_order_pairs(
        toy_corpus, table, per_ruler_pair=21, seed=1)
    emitted = set(map(tuple, meta[["ruler_i", "ruler_j"]]
                      .drop_duplicates().to_numpy()))
    forbidden = {("Esarhaddon", "Ashurbanipal"),
                 ("Ashurbanipal", "Esarhaddon")}
    assert not (emitted & forbidden)      # ZERO pairs across the overlap
    assert emitted == _disjoint_pairs_from_table(table)


def test_make_order_pairs_deterministic(toy_corpus, toy_ruler_table):
    a = make_order_pairs(toy_corpus, toy_ruler_table,
                         per_ruler_pair=21, seed=7)
    b = make_order_pairs(toy_corpus, toy_ruler_table,
                         per_ruler_pair=21, seed=7)
    c = make_order_pairs(toy_corpus, toy_ruler_table,
                         per_ruler_pair=21, seed=8)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[2].equals(b[2])
    assert not a[2]["doc_i"].equals(c[2]["doc_i"])

def test_hsic_is_double_centered():
    """Mutation guard: tr(KHLH), not tr(KHL).

    A singly-centered HSIC survived the old suite. Double centering is
    what makes HSIC zero for independent variables; with one H the
    statistic keeps a mean term and no longer vanishes.
    """
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(120, 1)), dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(120, 1)), dtype=torch.float64)
    indep = float(hsic_loss(x, y))
    dep = float(hsic_loss(x, x ** 2))
    assert indep < 0.02 and dep > 5 * max(indep, 1e-6)

    # hand-computed reference on a 4x4 problem, following the module's
    # stated convention: K_ij = exp(-d2 / (2 sigma^2)), sigma = median
    # heuristic, statistic tr(KHLH)/(n-1)^2
    n = 4
    xs = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    ys = torch.tensor([[0.0], [1.0], [4.0], [9.0]], dtype=torch.float64)

    def rbf(a, ref_sigma=None):
        d2 = (a - a.T) ** 2
        sig = core._median_sigma(d2) if ref_sigma is None else ref_sigma
        return torch.exp(-d2 / (2.0 * sig ** 2))

    K, L = rbf(xs), rbf(ys)
    H = torch.eye(n, dtype=torch.float64) - 1.0 / n
    ref = float(torch.trace(K @ H @ L @ H) / (n - 1) ** 2)
    got = float(hsic_loss(xs, ys))
    assert abs(got - ref) < 1e-9, (got, ref)


def test_variance_loss_hinges_on_std_not_var():
    """Mutation guard: swapping std for var passed the old suite.

    At std = 0.5 the hinge must read 0.5 below the floor, not 0.75.
    """
    s = torch.tensor([-0.5, 0.5], dtype=torch.float64)      # std = 0.5 (pop)
    got = float(variance_loss(s, floor=1.0))
    assert abs(got - 0.5) < 1e-6, got


def test_bt_offdiag_weight_is_honoured():
    """Mutation guard: a 100x change to lambda_offdiag passed before."""
    rng = np.random.default_rng(1)
    z_a = torch.tensor(rng.normal(size=(64, 6)), dtype=torch.float64)
    z_b = z_a + 0.01 * torch.tensor(rng.normal(size=(64, 6)),
                                    dtype=torch.float64)
    lo = float(bt_loss(z_a, z_b, lambda_offdiag=1e-3))
    hi = float(bt_loss(z_a, z_b, lambda_offdiag=1e-1))
    assert hi > lo                       # off-diag term actually weighted
    # and the weighting is linear in lambda: (hi - lo) tracks the ratio
    mid = float(bt_loss(z_a, z_b, lambda_offdiag=5e-2))
    assert lo < mid < hi

    # NUMERIC PIN — an ordering-only assertion survives a global rescale
    # of the off-diagonal term (a x100 mutation passed it). Recompute the
    # Barlow objective by hand and demand exact agreement.
    def reference(za, zb, lam):
        n = za.shape[0]
        # mirror the module's standardization exactly: sqrt(var + eps)
        a = (za - za.mean(0)) / torch.sqrt(za.var(0, unbiased=False) + 1e-6)
        b = (zb - zb.mean(0)) / torch.sqrt(zb.var(0, unbiased=False) + 1e-6)
        c = (a.T @ b) / n
        on = torch.diagonal(c)
        off = c - torch.diag_embed(on)
        return float(((on - 1.0) ** 2).sum() + lam * (off ** 2).sum())

    for lam in (1e-3, 5e-2, 1e-1):
        assert abs(float(bt_loss(z_a, z_b, lambda_offdiag=lam))
                   - reference(z_a, z_b, lam)) < 1e-8, lam


def test_touching_intervals_emit_no_pairs(toy_corpus):
    """SLA: touching (a1 == b0) is NOT disjoint — the real corpus has 4
    such ruler pairs, incl. Sennacherib/Esarhaddon (411 docs). Mutating
    `<` to `<=` passed the old suite."""
    docs = toy_corpus.copy()
    tbl = pd.DataFrame({"ruler": ["A", "B"], "t_min": [-700.0, -681.0],
                        "t_max": [-681.0, -670.0],      # touch at -681
                        "proxy": True, "n_docs": [5, 5]})
    d = pd.DataFrame({"doc_id": [f"x{i}" for i in range(10)],
                      "ruler": ["A"] * 5 + ["B"] * 5,
                      "t": [-700.0, -690, -685, -683, -681,
                            -681.0, -678, -675, -672, -670]})
    pairs, margins, meta = make_order_pairs(d, tbl, t_std=10.0,
                                            per_ruler_pair=5, seed=0)
    assert len(pairs) == 0, meta
