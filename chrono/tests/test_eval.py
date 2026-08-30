"""A5 — evaluation protocol on hand-made splits over the toy corpus.

Everything is checked against constructions whose rho is known exactly:
perfect scores (s = t) must hit +1 on every draw, anti-scores -1, the
placebo must straddle 0, and battery/coverage outputs are compared to
hand-computed values. No real artifacts, no torch, CPU-fast.
"""
import numpy as np
import pandas as pd
import pytest

from chrono.eval import (BATTERY_COLS, battery, coverage, gkf_rho,
                         mc_balanced_rho, mean_width, placebo_rho,
                         winkler_score)

N_DRAWS = 100


def _mc_split(corpus_df, n_draws=N_DRAWS, n_test=60, seed=7):
    """Hand-made mc-style split: each fold is one draw (SLA §3)."""
    g = np.random.default_rng(seed)
    ids = corpus_df["doc_id"].to_numpy()
    folds = []
    for _ in range(n_draws):
        test = set(g.choice(ids, size=n_test, replace=False))
        folds.append({"train": sorted(set(ids) - test),
                      "test": sorted(test)})
    return {"name": "mc_toy", "kind": "mc_balanced", "seed": seed,
            "folds": folds}


def _gkf_split(corpus_df):
    """One fold per ruler (grouped hold-out), rulers in sorted order."""
    ids = set(corpus_df["doc_id"])
    folds = []
    for _, sub in corpus_df.groupby("ruler"):
        test = sorted(sub["doc_id"])
        folds.append({"train": sorted(ids - set(test)), "test": test})
    return {"name": "gkf_toy", "kind": "gkf_ruler", "seed": 0,
            "folds": folds}


@pytest.fixture
def perfect_scores(toy_corpus):
    return pd.Series(toy_corpus["t"].to_numpy(dtype=float),
                     index=pd.Index(toy_corpus["doc_id"]))


@pytest.fixture
def mc_split(toy_corpus):
    return _mc_split(toy_corpus)


@pytest.fixture
def gkf_split(toy_corpus):
    return _gkf_split(toy_corpus)


# ------------------------------------------------------------------ mc


def test_mc_perfect_scores_rho_one(perfect_scores, toy_corpus, mc_split):
    rhos = mc_balanced_rho(perfect_scores, toy_corpus, mc_split)
    assert rhos.shape == (N_DRAWS,)
    assert np.allclose(rhos, 1.0)


def test_mc_anti_scores_rho_minus_one(perfect_scores, toy_corpus,
                                      mc_split):
    rhos = mc_balanced_rho(-perfect_scores, toy_corpus, mc_split)
    assert np.allclose(rhos, -1.0)


def test_mc_uses_test_docs_only(perfect_scores, toy_corpus, mc_split):
    # corrupting every TRAIN score of draw 0 must not move draw 0's rho
    corrupt = perfect_scores.copy()
    train0 = mc_split["folds"][0]["train"]
    corrupt.loc[train0] = -corrupt.loc[train0]
    rhos = mc_balanced_rho(corrupt, toy_corpus, mc_split)
    assert rhos[0] == pytest.approx(1.0)


def test_missing_doc_raises_named_error(perfect_scores, toy_corpus,
                                        mc_split):
    victim = mc_split["folds"][0]["test"][0]
    holed = perfect_scores.drop(victim)
    with pytest.raises(KeyError, match=victim):
        mc_balanced_rho(holed, toy_corpus, mc_split)


def test_doc_missing_from_corpus_raises(perfect_scores, toy_corpus,
                                        mc_split):
    victim = mc_split["folds"][0]["test"][0]
    holed = toy_corpus[toy_corpus["doc_id"] != victim]
    with pytest.raises(KeyError, match=victim):
        mc_balanced_rho(perfect_scores, holed, mc_split)


def test_bad_split_shape_raises(perfect_scores, toy_corpus):
    with pytest.raises(ValueError, match="folds"):
        mc_balanced_rho(perfect_scores, toy_corpus, {"name": "x"})


# ------------------------------------------------------------- placebo


def test_placebo_straddles_zero(perfect_scores, toy_corpus, mc_split):
    rhos = placebo_rho(perfect_scores, toy_corpus, mc_split, seed=11)
    assert rhos.shape == (N_DRAWS,)
    assert abs(rhos.mean()) < 0.1
    assert (rhos > 0).any() and (rhos < 0).any()


def test_placebo_deterministic_in_seed(perfect_scores, toy_corpus,
                                       mc_split):
    a = placebo_rho(perfect_scores, toy_corpus, mc_split, seed=11)
    b = placebo_rho(perfect_scores, toy_corpus, mc_split, seed=11)
    c = placebo_rho(perfect_scores, toy_corpus, mc_split, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------- gkf


def test_gkf_perfect_scores(perfect_scores, toy_corpus, gkf_split):
    by_fold = {k: perfect_scores.loc[f["test"]]
               for k, f in enumerate(gkf_split["folds"])}
    rhos = gkf_rho(by_fold, toy_corpus, gkf_split)
    assert rhos.shape == (len(gkf_split["folds"]),)
    assert np.allclose(rhos, 1.0)


def test_gkf_missing_fold_raises(perfect_scores, toy_corpus, gkf_split):
    by_fold = {k: perfect_scores.loc[f["test"]]
               for k, f in enumerate(gkf_split["folds"])}
    del by_fold[2]
    with pytest.raises(KeyError, match="fold"):
        gkf_rho(by_fold, toy_corpus, gkf_split)


# ------------------------------------------------------------- battery


def test_battery_grid_exact(perfect_scores, toy_corpus, mc_split,
                            gkf_split):
    scores_df = pd.concat([
        pd.DataFrame({"doc_id": perfect_scores.index, "condition": "orig",
                      "s": perfect_scores.to_numpy()}),
        pd.DataFrame({"doc_id": perfect_scores.index, "condition": "anti",
                      "s": -perfect_scores.to_numpy()}),
    ], ignore_index=True)
    splits = {"mc_balanced": mc_split, "gkf_ruler": gkf_split}
    out = battery(scores_df, toy_corpus, splits)

    assert list(out.columns) == BATTERY_COLS == [
        "condition", "split", "rho_mean", "rho_sd", "n"]
    assert len(out) == 4                        # 2 conditions x 2 splits
    assert list(out["condition"]) == ["orig", "orig", "anti", "anti"]
    assert list(out["split"]) == ["mc_balanced", "gkf_ruler"] * 2

    cell = out.set_index(["condition", "split"])
    assert cell.loc[("orig", "mc_balanced"), "rho_mean"] == \
        pytest.approx(1.0)
    assert cell.loc[("anti", "gkf_ruler"), "rho_mean"] == \
        pytest.approx(-1.0)
    assert cell.loc[("orig", "mc_balanced"), "rho_sd"] == \
        pytest.approx(0.0)
    assert (out.loc[out["split"] == "mc_balanced", "n"] == N_DRAWS).all()
    assert (out.loc[out["split"] == "gkf_ruler", "n"]
            == len(gkf_split["folds"])).all()


def test_battery_missing_doc_raises(perfect_scores, toy_corpus,
                                    mc_split):
    scores_df = pd.DataFrame(
        {"doc_id": perfect_scores.index, "condition": "orig",
         "s": perfect_scores.to_numpy()}).iloc[:-1]     # drop one doc
    dropped = perfect_scores.index[-1]
    if dropped not in mc_split["folds"][0]["test"]:     # ensure it's hit
        mc_split["folds"][0]["test"].append(dropped)
    with pytest.raises(KeyError, match="scores"):
        battery(scores_df, toy_corpus, {"mc": mc_split})


def test_battery_rejects_bad_frames(perfect_scores, toy_corpus,
                                    mc_split):
    with pytest.raises(ValueError, match="missing column"):
        battery(pd.DataFrame({"doc_id": [], "s": []}), toy_corpus,
                {"mc": mc_split})
    twice = pd.DataFrame({"doc_id": ["D0_0", "D0_0"],
                          "condition": ["orig", "orig"], "s": [1., 2.]})
    with pytest.raises(ValueError, match="duplicate"):
        battery(twice, toy_corpus, {"mc": mc_split})


# --------------------------------------------------------- calibration


def test_coverage_hand_computed():
    lo, hi = [0., 0., 0., 0.], [2., 2., 2., 2.]
    t = [1.0, 3.0, 2.0, -2.0]        # in, above, on-bound (in), below
    assert coverage(lo, hi, t) == pytest.approx(0.5)
    assert coverage([-1.], [1.], [0.]) == pytest.approx(1.0)


def test_interval_metrics_hand_computed():
    lo, hi = [0., 0., 0., 0.], [2., 2., 2., 2.]
    t = [1.0, 3.0, 2.0, -2.0]
    assert mean_width(lo, hi) == pytest.approx(2.0)
    # alpha=.2: widths 2 each; misses 0, 1, 0, 2 -> 2+12+2+22 over 4
    assert winkler_score(lo, hi, t, nominal=0.8) == pytest.approx(9.5)


def test_calibration_validation():
    with pytest.raises(ValueError, match="hi < lo"):
        coverage([1.0], [0.0], [0.5])
    with pytest.raises(ValueError, match="shape"):
        coverage([0.0, 0.0], [1.0, 1.0], [0.5])
    with pytest.raises(ValueError, match="nominal"):
        winkler_score([0.0], [1.0], [0.5], nominal=1.0)
