"""E1 pair engine: balanced Monte-Carlo draws of "which fragment is earlier?" pairs.

THE IMBALANCE THIS SOLVES. Fragment counts per ruler are wildly skewed (268 for
Ashurbanipal down to 1 for twelve tail rulers), and pairing SQUARES the skew:
Ashurbanipal x Sennacherib alone is 63,516 possible pairs while a tail x tail
ruler-pair has exactly one. Unbalanced training would be dominated by four rulers.

THE BALANCING UNIT IS THE RULER-PAIR, mirroring how the regression design's unit
is the ruler. Per Monte-Carlo draw, every eligible ruler-pair contributes
min(m, n_i, n_j) fragment pairs, sampled fresh each draw with NO fragment reused
within the same ruler-pair in the same draw. Because every ruler sits in ~39
ruler-pairs, an equal quota per ruler-pair also equalizes rulers automatically.

Residual imbalance (a 1-fragment ruler can only ever contribute 1 pair vs m from
the giants) is handled downstream, twice:
  * training: sample_weight = 1/m_ij, so every ruler-pair carries equal total loss;
  * evaluation: metrics are macro-averaged over ruler-pairs, never over raw pairs.

Data exhaustion happens ACROSS draws: each draw resamples the big grids, so over
100 draws a 63k-pair grid is covered ~3% per draw with fresh pairs each time; the
--m 100 robustness setting pushes coverage further. Exhausting literally every
pair in one shot would just recreate the imbalance this module exists to remove.

Eligibility: fragments with a year label (drops 9 undated rows plus the entire
'ribo' pseudo-ruler, whose 9 rows are all undated); ruler-pairs where at least one
cross pair has differing years. Same-year fragment pairs carry no order and are
never emitted. Ruler-pairs whose year ranges guarantee identical years (i.e. two
single-year rulers sharing the year) are dropped entirely.

Self-test (CPU, seconds):    python pairs_data.py
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM_AKK = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models", "akkadian"))
sys.path.insert(0, _WM_AKK)
sys.path.insert(0, os.path.dirname(_WM_AKK))
import akk_data as A                                   # noqa: E402

M_DEFAULT = 21          # echoes the regression design's k=21 per-ruler cap
N_DRAWS_DEFAULT = 100
N_FOLDS = 5
SEED = 42


def load_eligible() -> pd.DataFrame:
    """Dated fragments only, with a positional index into the activation arrays.

    `akk_data.load_fragments()` is exactly the frame the activation extractors
    iterated over, so `pos` below indexes rows of every {site}.layer{L}.npz.
    """
    df = A.load_fragments().reset_index(drop=True)
    df["pos"] = np.arange(len(df))
    df = df[df.year.notna() & df.ruler.notna()].copy()
    df["year"] = df["year"].astype(float)
    return df


def eligible_ruler_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Ruler-pairs that can produce at least one ordered (different-year) pair."""
    years = df.groupby("ruler")["year"].agg(["min", "max"])
    out = []
    for ra, rb in combinations(sorted(years.index), 2):
        a, b = years.loc[ra], years.loc[rb]
        # a pair is hopeless only if both rulers are single-year AND share it
        if a["min"] == a["max"] == b["min"] == b["max"]:
            continue
        out.append((ra, rb))
    return out


def draw_pairs(df: pd.DataFrame, m: int, rng: np.random.Generator,
               ruler_pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """One Monte-Carlo draw: a frame of ordered pairs, balanced per ruler-pair.

    Columns: pos_a, pos_b, ruler_a, ruler_b, year_a, year_b, label (1 = a earlier),
    weight (1/m_ij), dyear (|year_a - year_b|). The presented order (a, b) is
    randomized per pair so the label is ~50/50 within every ruler-pair.
    """
    if ruler_pairs is None:
        ruler_pairs = eligible_ruler_pairs(df)
    by_ruler = {r: g for r, g in df.groupby("ruler")}
    rows = []
    for ra, rb in ruler_pairs:
        ga, gb = by_ruler[ra], by_ruler[rb]
        k = min(m, len(ga), len(gb))
        ia = rng.choice(len(ga), size=k, replace=False)
        ib = rng.choice(len(gb), size=k, replace=False)
        sub = pd.DataFrame({
            "pos_a": ga["pos"].values[ia], "pos_b": gb["pos"].values[ib],
            "year_a": ga["year"].values[ia], "year_b": gb["year"].values[ib],
            "ruler_a": ra, "ruler_b": rb,
        })
        sub = sub[sub.year_a != sub.year_b]        # same-year pairs carry no order
        if not len(sub):
            continue
        # randomize presentation order so "earlier" is not encoded by position
        flip = rng.random(len(sub)) < 0.5
        for c in ("pos", "year", "ruler"):
            av, bv = sub[f"{c}_a"].copy(), sub[f"{c}_b"].copy()
            sub[f"{c}_a"] = np.where(flip, bv, av)
            sub[f"{c}_b"] = np.where(flip, av, bv)
        sub["label"] = (sub.year_a < sub.year_b).astype(int)
        sub["weight"] = 1.0 / len(sub)             # equal total weight per ruler-pair
        sub["dyear"] = (sub.year_a - sub.year_b).abs()
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def draw_within_ruler(df: pd.DataFrame, ruler: str, n_pairs: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Ordered pairs INSIDE one ruler (E6): both fragments share the ruler, so
    any order signal is identity-free by construction. Only rulers with
    within-ruler year variance produce anything (Esarhaddon: 176 frags, 11 yrs).
    Same output columns as draw_pairs."""
    g = df[df.ruler == ruler]
    ia = rng.integers(0, len(g), n_pairs * 3)
    ib = rng.integers(0, len(g), n_pairs * 3)
    ya, yb = g.year.values[ia], g.year.values[ib]
    keep = ya != yb
    ia, ib, ya, yb = ia[keep][:n_pairs], ib[keep][:n_pairs], \
        ya[keep][:n_pairs], yb[keep][:n_pairs]
    return pd.DataFrame({
        "pos_a": g.pos.values[ia], "pos_b": g.pos.values[ib],
        "year_a": ya, "year_b": yb, "ruler_a": ruler, "ruler_b": ruler,
        "label": (ya < yb).astype(int), "weight": 1.0,
        "dyear": np.abs(ya - yb)})


def ruler_folds(rulers: list[str], rng: np.random.Generator,
                n_folds: int = N_FOLDS) -> dict[str, int]:
    """Random ruler -> fold assignment, reshuffled per draw.

    A pair trains only if BOTH rulers are outside the test fold and tests only if
    BOTH are inside it: the pairwise analog of GroupKFold-by-ruler, which is the
    protocol that killed the leak in the regression design. Pairs straddling
    train/test are used for neither.
    """
    order = list(rulers)
    rng.shuffle(order)
    return {r: i % n_folds for i, r in enumerate(order)}


def _selftest():
    df = load_eligible()
    rp = eligible_ruler_pairs(df)
    print(f"eligible fragments: {len(df)} | rulers: {df.ruler.nunique()} "
          f"| ruler-pairs: {len(rp)}")
    rng = np.random.default_rng(SEED)
    d = draw_pairs(df, M_DEFAULT, rng, rp)
    canon = [d[["ruler_a", "ruler_b"]].min(axis=1),
             d[["ruler_a", "ruler_b"]].max(axis=1)]
    print(f"one draw (m={M_DEFAULT}): {len(d)} pairs, "
          f"label balance {d.label.mean():.3f}, "
          f"ruler-pairs represented {d.groupby(canon).ngroups}")
    counts = d.groupby(canon).size()
    print(f"pairs per ruler-pair: min {counts.min()}, median {counts.median():.0f}, "
          f"max {counts.max()}  (max should be <= m)")
    w = d.groupby([d[["ruler_a", "ruler_b"]].min(axis=1),
                   d[["ruler_a", "ruler_b"]].max(axis=1)])["weight"].sum()
    assert np.allclose(w.values, 1.0), "per-ruler-pair total weight must be 1"
    folds = ruler_folds(sorted(df.ruler.unique()), rng)
    both_in = [f for f in range(N_FOLDS)
               for _ in range(sum(folds[a] == folds[b] == f
                                  for a, b in zip(d.ruler_a, d.ruler_b)))]
    print(f"testable pairs under one 5-fold split: {len(both_in)} "
          f"({len(both_in) / len(d):.1%} of the draw; the rest re-enter on later "
          "draws when the reshuffled folds co-locate their rulers)")
    # label must not be recoverable from presentation order
    assert 0.4 < d.label.mean() < 0.6
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
