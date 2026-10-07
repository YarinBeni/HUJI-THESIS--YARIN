"""EmbStore under concurrent writers (CLUSTER FIX, C1v2 jobs 33341-33344):
several processes put into ONE root at once; the manifest must end up with
every row, and a corrupt manifest must rebuild itself from the shards."""
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from chrono.models.store import EmbStore  # noqa: E402


def _writer(args):
    root, model, n_puts = args
    st = EmbStore(root)
    rng = np.random.default_rng(hash(model) % 2**32)
    for k in range(n_puts):
        ids = [f"{model}-{k}-{i}" for i in range(50)]
        st.put(model, k % 3, "mean", ids, rng.normal(size=(50, 8)),
               texts=[f"t{i}" for i in range(50)])
    return n_puts


def test_concurrent_puts_keep_every_row(tmp_path):
    root = str(tmp_path / "store")
    models = [f"m{j}" for j in range(6)]
    with mp.get_context("spawn").Pool(6) as pool:
        pool.map(_writer, [(root, m, 12) for m in models])
    st = EmbStore(root)
    m = st.manifest()
    assert len(m) == 6 * 12 * 50, len(m)
    assert not m.duplicated(["id", "model", "layer", "site"]).any()
    X = st.get("m3", 1, "mean", [f"m3-1-{i}" for i in range(50)],
               texts=[f"t{i}" for i in range(50)])
    assert X.shape == (50, 8)


def test_corrupt_manifest_rebuilds_from_shards(tmp_path):
    root = str(tmp_path / "store")
    st = EmbStore(root)
    ids = [f"d{i}" for i in range(20)]
    X = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    st.put("Org/Model", 5, "last", ids, X, texts=ids)
    with open(st.manifest_path, "wb") as f:      # truncate -> unreadable
        f.write(b"PAR1 garbage")
    st2 = EmbStore(root)
    got = st2.get("Org/Model", 5, "last", ids, texts=ids)
    np.testing.assert_array_equal(got, X)
    assert st2.manifest()["text_sha"].ne("").all()


def test_manifest_is_never_written_in_place(tmp_path):
    root = str(tmp_path / "store")
    st = EmbStore(root)
    st.put("m", 0, "mean", ["a", "b"], np.zeros((2, 3), np.float32))
    leftovers = [f for f in os.listdir(root) if f.startswith(".manifest.")
                 and f.endswith(".parquet")]
    assert leftovers == []                        # temp file replaced, not left
