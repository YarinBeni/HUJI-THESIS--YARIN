"""A4 tests: EmbStore round-trip, EMA twin math, and the P3.1 overfit
gate — 50 synthetic docs with a monotone tfidf-able signal must reach
train soft-Spearman >= .95 within 200 CPU epochs (SLA section 6)."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from chrono import common
from chrono.models.heads import AdapterHead, EmaTwin
from chrono.models.store import EmbStore
from chrono.scripts import train_cjb

CFG_DIR = os.path.join(common.CHRONO, "configs")


# --------------------------------------------------------------- store
def _put_block(store, n=7, d=5, seed=0, ids=None, **key):
    key = {"model": key.get("model", "toy/enc"),
           "layer": key.get("layer", 3), "site": key.get("site", "mean")}
    ids = ids or [f"v{i}" for i in range(n)]
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    store.put(key["model"], key["layer"], key["site"], ids, X,
              texts=[f"text {i}" for i in ids])
    return key, ids, X


def test_embstore_roundtrip(tmp_path):
    store = EmbStore(tmp_path / "store")
    key, ids, X = _put_block(store)
    got = store.get(key["model"], key["layer"], key["site"], ids)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, X)
    # order follows the caller's ids, not insertion
    rev = store.get(key["model"], key["layer"], key["site"], ids[::-1])
    np.testing.assert_array_equal(rev, X[::-1])
    # has(): elementwise membership
    flags = store.has(key["model"], key["layer"], key["site"],
                      [ids[0], "nope", ids[-1]])
    assert flags.tolist() == [True, False, True]
    # a different (model, layer, site) knows nothing
    assert not store.has("other", 0, "last", ids).any()
    # manifest schema is the contract's, exactly
    assert list(store.manifest().columns) == \
        ["id", "model", "layer", "site", "dim", "shard", "row",
         "text_sha"]
    assert (store.manifest()["text_sha"] != "").all()


def test_embstore_missing_raises_with_ids(tmp_path):
    store = EmbStore(tmp_path / "store")
    key, ids, _ = _put_block(store)
    with pytest.raises(KeyError) as e:
        store.get(key["model"], key["layer"], key["site"],
                  [ids[0], "ghost1", "ghost2"])
    assert "ghost1" in str(e.value) and "ghost2" in str(e.value)
    assert ids[0] not in str(e.value)


def test_embstore_deterministic_shards_and_overwrite(tmp_path):
    s1, s2 = EmbStore(tmp_path / "a"), EmbStore(tmp_path / "b")
    _put_block(s1)
    _put_block(s2)
    assert sorted(os.listdir(s1.root)) == sorted(os.listdir(s2.root))
    # re-putting the same ids replaces, not duplicates
    key, ids, _ = _put_block(s1, seed=1)
    m = s1.manifest()
    assert len(m) == len(ids)
    X2 = s1.get(key["model"], key["layer"], key["site"], ids)
    np.testing.assert_array_equal(
        X2, np.random.default_rng(1).normal(
            size=(7, 5)).astype(np.float32))


# --------------------------------------------------------------- heads
def test_adapter_head_shapes():
    torch.manual_seed(0)
    head = AdapterHead(d_in=10, d_hidden=16, d_proj=4)
    h, s, p = head(torch.randn(8, 10))
    assert h.shape == (8, 16) and s.shape == (8,) and p.shape == (8, 4)
    assert s.requires_grad


def test_ema_twin_update_math_and_stopgrad():
    torch.manual_seed(0)
    head = AdapterHead(d_in=6, d_hidden=8, d_proj=3)
    m = 0.9
    twin = EmaTwin(head, momentum=m)
    xi0 = [p.detach().clone() for p in twin.target.parameters()]
    with torch.no_grad():
        for p in head.parameters():
            p.add_(1.0)
    twin.update()
    for xi, old, theta in zip(twin.target.parameters(), xi0,
                              head.parameters()):
        torch.testing.assert_close(xi, m * old + (1 - m) * theta,
                                   rtol=0, atol=1e-6)
    # stop-grad: target params never require grad, forward carries none
    assert all(not p.requires_grad for p in twin.target.parameters())
    h, s, p = twin(torch.randn(4, 6))
    assert h.grad_fn is None and s.grad_fn is None and p.grad_fn is None
    # optimizer surface excludes the online head
    n_target = sum(1 for _ in twin.target.parameters())
    assert sum(1 for _ in twin.parameters()) == n_target


# ------------------------------------------------------------- trainer
def _synth(n=50, docs_per_ruler=5):
    """Docs with a tfidf-able monotone signal: the marker word count
    grows with t; rulers tile t into disjoint reign-proxy intervals."""
    rows, views = [], []
    for i in range(n):
        k = i // docs_per_ruler
        text = ("the king built the wall of the temple "
                + "glorious " * (2 + i)).strip()
        rows.append(dict(doc_id=f"S{i}", ruler=f"R{k:02d}", t=float(i),
                         text_eng=text, text_akk=""))
        views.append(dict(view_id=f"S{i}::eng::+s0", doc_id=f"S{i}",
                          lang="eng", augs="", seed=0, text=text,
                          n_words=len(text.split()), mask_count=0))
    corpus = pd.DataFrame(rows)
    g = corpus.groupby("ruler")["t"]
    ruler_table = pd.DataFrame({
        "ruler": g.min().index, "t_min": g.min().values,
        "t_max": g.max().values, "proxy": True,
        "n_docs": g.size().values})
    return corpus, pd.DataFrame(views), ruler_table


def _cfg(epochs, lr=0.02, batch=64, seed=0):
    return {
        "run_name": "unit",
        "features": {"kind": "tfidf", "model": None, "layer": None,
                     "site": None},
        "views": {"menu_a": [[]], "menu_b": [[]], "seeds": [0]},
        "loss": {"lambda_rank": 1.0, "lambda_var": 0.1, "temp": 0.5,
                 "lambda_offdiag": 5e-3},
        "train": {"epochs": epochs, "lr": lr, "batch": batch,
                  "seed": seed},
        "eval_split": "",
    }


def test_overfit_gate_tfidf():
    corpus, views, ruler_table = _synth()
    res = train_cjb.train(_cfg(epochs=200), corpus, views,
                          ruler_table=ruler_table, write=False,
                          log_every=0)
    assert res["metrics"]["train_soft_spearman"] >= 0.95, res["metrics"]
    assert res["metrics"]["train_spearman"] >= 0.95
    assert len(res["loss_curve"]) == 200
    assert np.isfinite(res["loss_curve"]).all()
    # scores frame carries the SLA schema
    assert list(res["scores"].columns) == \
        ["run_id", "doc_id", "condition", "s", "fit", "fold",
         "is_test", "s_rank"]
    assert (res["scores"]["fit"] == "full").all()      # no fold given
    # review fix: every augmentation chain is scored, not just 'orig'
    assert "orig" in set(res["scores"]["condition"])
    assert (res["scores"]["condition"] == "orig").all()
    n_cond = res["scores"]["condition"].nunique()
    assert len(res["scores"]) == len(corpus) * n_cond


def test_trainer_emb_path(tmp_path):
    corpus, views, ruler_table = _synth(n=30, docs_per_ruler=5)
    store = EmbStore(tmp_path / "store")
    rng = np.random.default_rng(0)
    t = corpus["t"].to_numpy()
    X = np.c_[t / 10.0, rng.normal(size=(len(t), 7))] \
        .astype(np.float32)                      # dim 0 IS lateness
    store.put("toy/enc", 0, "mean", views["view_id"].tolist(), X)
    cfg = _cfg(epochs=120, lr=0.02)
    cfg["features"] = {"kind": "emb", "model": "toy/enc", "layer": 0,
                       "site": "mean"}
    res = train_cjb.train(cfg, corpus, views, store=store,
                          ruler_table=ruler_table, write=False,
                          log_every=0)
    assert res["metrics"]["train_spearman"] >= 0.9, res["metrics"]


def test_trainer_writes_scores_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ART", str(tmp_path))
    corpus, views, ruler_table = _synth(n=20, docs_per_ruler=5)
    res = train_cjb.train(_cfg(epochs=3), corpus, views,
                          ruler_table=ruler_table, write=True,
                          out_dir=str(tmp_path / "scores"), log_every=0)
    got = pd.read_parquet(res["scores_path"])
    assert list(got.columns) == ["run_id", "doc_id", "condition", "s",
                                 "fit", "fold", "is_test", "s_rank"]
    assert set(got["doc_id"]) == set(corpus["doc_id"])
    assert set(got.loc[got["condition"] == "orig", "doc_id"]) == \
        set(corpus["doc_id"])
    results = pd.read_parquet(tmp_path / "results.parquet")
    assert list(results.columns) == common.RESULTS_COLS
    assert "train_soft_spearman" in set(results["metric"])
    assert (results["run_id"] == res["run_id"]).all()


def test_trainer_deterministic():
    corpus, views, ruler_table = _synth(n=20, docs_per_ruler=5)
    r1 = train_cjb.train(_cfg(epochs=5), corpus, views,
                         ruler_table=ruler_table, write=False,
                         log_every=0)
    r2 = train_cjb.train(_cfg(epochs=5), corpus, views,
                         ruler_table=ruler_table, write=False,
                         log_every=0)
    np.testing.assert_array_equal(r1["scores"]["s"], r2["scores"]["s"])
    assert r1["loss_curve"] == r2["loss_curve"]


def test_config_sha_stability():
    for name in ("emin_tfidf_smoke.yaml", "emin_thalesian.yaml"):
        path = os.path.join(CFG_DIR, name)
        c1, c2 = train_cjb.load_config(path), train_cjb.load_config(path)
        assert c1 == c2
        assert common.config_sha(c1) == common.config_sha(c2)
        assert set(c1) == train_cjb.CONFIG_KEYS
    smoke = train_cjb.load_config(
        os.path.join(CFG_DIR, "emin_tfidf_smoke.yaml"))
    thal = train_cjb.load_config(
        os.path.join(CFG_DIR, "emin_thalesian.yaml"))
    assert common.config_sha(smoke) != common.config_sha(thal)
    bumped = {**smoke, "train": {**smoke["train"], "seed": 99}}
    assert common.config_sha(bumped) != common.config_sha(smoke)
