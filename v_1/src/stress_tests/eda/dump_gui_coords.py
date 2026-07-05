"""dump_gui_coords.py — 2-D coordinates of the NEW embeddings for the seal_eda GUI.

For every model with maximal-with-kings activations on disk, take the maxking-mean
matrix at that model's best maxking layer (from p1_maxking results; falls back to
L0) plus L0, and compute PCA-2D + t-SNE-2D over the 1,202 ORCC fragments. Also dump
the 153 P3 anchors (kind/ruler/year) at the same layers so they can be overlaid.

Output (committed; small): v_1/src/viz/maxking_coords.json
  {"fragment_ids": [...], "embeddings": {"maxking__<method>__L<NN>__pca": [[x,y]..],
   "maxking__<method>__L<NN>__tsne": ...}, "anchors": {...}}
The GUI merge (02_merge_coords.py-style) is done off-cluster after this lands.

Usage:  python v_1/src/stress_tests/eda/dump_gui_coords.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402

ST = _REPO / "v_1/src/stress_tests"
CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = _REPO / "v_1/src/viz/maxking_coords.json"
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "random"]


def best_layer(method):
    fp = ST / f"p1_gurnee_tegmark/results/maxking/p1_maxking__{method}.json"
    if fp.exists():
        d = json.loads(fp.read_text())
        s = d.get("sites", {}).get("mean", {})
        if "best_layer" in s:
            return int(s["best_layer"])
    return 0


def coords2d(X):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Xz = np.nan_to_num(X.astype(np.float64))
    p = PCA(n_components=2, random_state=42).fit_transform(Xz)
    t = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(Xz)
    return (np.round(p, 4).tolist(), np.round(t, 4).tolist())


def main():
    df = pd.read_parquet(CORPUS)
    out = {"fragment_ids": df["fragment_id"].astype(str).tolist(),
           "embeddings": {}, "anchors": {}}
    for m in MODELS:
        d = find_acts_dir(m, "maxking", "mean")
        if d is None:
            print(f"skip {m} (no maxking acts)"); continue
        layers = sorted({0, best_layer(m)} & set(available_layers(d)))
        for L in layers:
            pca, tsne = coords2d(load_layer(d, L))
            out["embeddings"][f"maxking__{m}__L{L:02d}__pca"] = pca
            out["embeddings"][f"maxking__{m}__L{L:02d}__tsne"] = tsne
            print(f"{m} L{L}: done")
        # anchors at the best layer (for timeline overlay)
        adir = ST / f"p3_matter_of_time/anchors/{m}"
        npz = adir / f"L{layers[-1]:02d}.npz"
        if npz.exists():
            a = np.load(npz, allow_pickle=True)
            pca, tsne = coords2d(a["acts"])
            items = json.loads((adir / "anchors.json").read_text())
            out["anchors"][f"{m}__L{layers[-1]:02d}"] = {
                "pca": pca, "tsne": tsne,
                "years": [it["year"] for it in items],
                "kinds": [it["kind"] for it in items],
                "rulers": [it.get("ruler") for it in items]}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
