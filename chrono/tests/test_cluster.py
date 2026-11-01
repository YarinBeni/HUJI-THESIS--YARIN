"""A6 tests — extraction selftest, baseline gate on synthetic features,
sbatch syntax. All CPU, no transformers, no network: the selftest path
uses the in-file stub encoder, the gate runs on a planted linear-t
signal in a throwaway EmbStore, and the sbatch files only get bash -n
(the cluster semantics are exercised by C0 itself)."""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "chrono", "scripts")
SBATCH = os.path.join(REPO, "chrono", "sbatch")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"chrono_scripts_{name}", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# extract_embeddings

def test_extract_selftest_end_to_end(tmp_path):
    root = str(tmp_path / "store")
    r = subprocess.run(
        [sys.executable,
         os.path.join(SCRIPTS, "extract_embeddings.py"),
         "--selftest", "--store-root", root],
        capture_output=True, text=True, cwd=REPO, timeout=240)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[selftest] OK" in r.stdout
    m = pd.read_parquet(os.path.join(root, "manifest.parquet"))
    # 5 texts x 3 stub layers x 2 sites, each id exactly once per cell
    assert len(m) == 30
    assert set(m["site"]) == {"mean", "last"}
    assert set(m["layer"]) == {0, 1, 2}
    assert not m.duplicated(["id", "model", "layer", "site"]).any()


def test_gather_texts_id_scheme(toy_corpus):
    ee = _load("extract_embeddings")
    views = pd.DataFrame({
        "view_id": ["D0_0::eng::mask_ruler+s0", "D0_1::akk::+s0"],
        "text": ["<RULER> built the wall", "sar mat assur"]})
    table = ee.gather_texts(views, toy_corpus)
    ids = set(table["id"])
    assert "D0_0::eng::mask_ruler+s0" in ids
    assert "D0_0::akk::orig" in ids and "D0_0::eng::orig" in ids
    assert len(table) == 2 + 2 * len(toy_corpus)
    assert table["id"].is_monotonic_increasing  # stable chunking
    assert ee.parse_layers("0-3,7") == [0, 1, 2, 3, 7]


# --------------------------------------------------------------------------
# run_baseline_gate

def _synthetic_world(tmp_path, toy_corpus):
    """EmbStore with a planted linear-t signal at layer 1 (layer 0 is
    pure noise) + gkf/mc split JSONs in the SLA section 3 shape."""
    from sklearn.model_selection import GroupKFold

    from chrono.models.store import EmbStore

    rng = np.random.default_rng(0)
    docs = toy_corpus["doc_id"].to_numpy()
    t = toy_corpus["t"].to_numpy(dtype=float)
    ids = [f"{d}::akk::orig" for d in docs]
    store = EmbStore(str(tmp_path / "store"))
    for layer in (0, 1):
        X = rng.standard_normal((len(docs), 16)).astype(np.float32)
        if layer == 1:
            X[:, 0] = (t - t.mean()) / t.std() \
                + 0.1 * rng.standard_normal(len(docs))
        store.put("stub", layer, "mean", ids, X)

    folds = []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(docs, groups=toy_corpus["ruler"]):
        folds.append({"train": sorted(docs[tr].tolist()),
                      "test": sorted(docs[te].tolist())})
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    with open(splits_dir / "gkf_ruler.json", "w") as f:
        json.dump({"name": "gkf_ruler", "kind": "gkf", "seed": 0,
                   "folds": folds}, f)

    rulers = toy_corpus["ruler"].unique()
    draws = []
    for _ in range(30):
        pick = rng.choice(rulers, size=4, replace=False)
        test = []
        for ru in pick:
            sub = toy_corpus[toy_corpus["ruler"] == ru]["doc_id"]
            test += rng.choice(sub, size=10, replace=False).tolist()
        draws.append({"train": sorted(set(docs) - set(test)),
                      "test": sorted(test)})
    with open(splits_dir / "mc_balanced.json", "w") as f:
        json.dump({"name": "mc_balanced", "kind": "mc", "seed": 0,
                   "folds": draws}, f)
    return store.root, str(splits_dir)


def test_baseline_gate_synthetic(tmp_path, toy_corpus, monkeypatch):
    from chrono import common

    store_root, splits_dir = _synthetic_world(tmp_path, toy_corpus)
    corpus_path = str(tmp_path / "corpus.parquet")
    toy_corpus.to_parquet(corpus_path, index=False)
    art = tmp_path / "art"
    art.mkdir()
    monkeypatch.setattr(common, "ART", str(art))  # results land in tmp
    report = str(tmp_path / "gate_report.txt")

    gate = _load("run_baseline_gate")
    gate.main([
        "--model", "stub", "--layers", "0-1", "--sites", "mean",
        "--lang", "akk", "--probes", "ridge", "pls",
        "--corpus", corpus_path, "--splits-dir", splits_dir,
        "--store-root", store_root, "--report-out", report,
        "--seed", "0", "--gate-rho", "0.2"])

    res = pd.read_parquet(art / "results.parquet")
    # 2 layers x 2 probes x (2 gkf + 2 mc + 1 placebo) rows
    assert len(res) == 20
    assert res["run_id"].str.startswith("p04_gate::stub::").all()
    assert set(res["split"]) == {"gkf_ruler", "mc_balanced"}

    def cell(layer, probe, metric):
        m = res[(res["run_id"] == f"p04_gate::stub::L{layer}::mean"
                 f"::{probe}") & (res["metric"] == metric)
                & (res["split"] == "mc_balanced")]
        assert len(m) == 1
        return float(m["value"].iloc[0])

    # planted signal recovered at L1, absent at L0, placebo near zero
    for probe in ("ridge", "pls"):
        assert cell(1, probe, "rho_mean") > 0.6
        assert abs(cell(0, probe, "rho_mean")) < 0.4
        assert abs(cell(1, probe, "placebo_rho_mean")) < 0.3

    txt = open(report).read()
    assert "P0.4 BASELINE GATE" in txt
    # review fix: the verdict is now the a-priori cell; the best cell is
    # printed only as selection-inflated context
    assert "VERDICT CELL (" in txt
    assert "SELECTION-INFLATED" in txt
    assert "block null" in txt
    assert "verdict: PASS" in txt          # 0.6+ vs --gate-rho 0.2
    assert "re-pin" in txt                 # above band -> re-pin note


def test_apriori_cell_is_a_real_layer():
    """The verdict cell must exist in the encoder we actually run.
    Thalesian/AKK_300m returns 9 hidden states (0..8); an a-priori layer
    outside that range makes verdict_block fall through to the
    selection-inflated best cell without saying so."""
    gate = _load("run_baseline_gate")
    assert gate.APRIORI_LAYER in range(0, 9), gate.APRIORI_LAYER
    assert gate.APRIORI_LAYER in gate._parse_layers("0-8")


def test_baseline_gate_unpinned_verdict():
    gate = _load("run_baseline_gate")
    rows = [dict(probe="ridge", layer=11, site="mean",
                 gkf=np.array([0.4, 0.42]), mc=np.array([0.41, 0.4]),
                 placebo=np.array([0.01, -0.02]))]
    txt = gate.verdict_block(rows, None, 0.02)
    assert "UNPINNED" in txt
    assert "v_1/src/phase2/pairs/RESULTS.md" in txt
    txt = gate.verdict_block(rows, 0.41, 0.02)
    assert "verdict: PASS" in txt
    txt = gate.verdict_block(rows, 0.60, 0.02)
    assert "verdict: FAIL" in txt


# --------------------------------------------------------------------------
# sbatch suite

def test_sbatch_files_exist_and_parse():
    names = ["_sandbox.sh", "C0_tests.sbatch", "C1_extract.sbatch",
             "C2_baseline_gate.sbatch", "C3_emin.sbatch"]
    for name in names:
        path = os.path.join(SBATCH, name)
        assert os.path.exists(path), f"missing {path}"
        r = subprocess.run(["bash", "-n", path], capture_output=True,
                           text=True, timeout=30)
        assert r.returncode == 0, f"{name}: {r.stderr}"


def test_sbatch_conventions():
    """The invariants the SLA cares about: sandbox sync (never main),
    gitignored artifacts copied into tracked chrono/reports/."""
    sandbox = open(os.path.join(SBATCH, "_sandbox.sh")).read()
    assert "yarin-sandbox" in sandbox
    assert "flock" in sandbox and "--autostash" in sandbox
    assert "HEAD:main" not in sandbox
    for name in ("C1_extract.sbatch", "C2_baseline_gate.sbatch",
                 "C3_emin.sbatch"):
        txt = open(os.path.join(SBATCH, name)).read()
        assert "source chrono/sbatch/_sandbox.sh" in txt
        assert "sync_sandbox" in txt
        assert "commit_push_sandbox" in txt
        assert "chrono/reports" in txt
    c0 = open(os.path.join(SBATCH, "C0_tests.sbatch")).read()
    assert "pytest chrono/tests" in c0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
