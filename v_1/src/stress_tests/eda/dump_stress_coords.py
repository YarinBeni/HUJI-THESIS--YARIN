"""dump_stress_coords.py — 2-D coords of the stress-test embeddings for the GUI.

Extends the J16 maxking dump to the cleanings the stress tests added:
  * maximal   — name-stripped Akkadian (all 8 models + mlm + random)
  * engtier0  — English translations of tier0 (7 models + random)
  * engmaximal — English translations of maximal (same; the GUI marks the
                 known caveat: the translator hallucinates king names)

Per model x cleaning: L0 + that model's best year layer (maximal: from the
committed p1_year_mc.csv; eng*: argmax of the per-layer year Spearman in
translation/results/trans_mc__<m>.json), PCA-2D + t-SNE-2D over the 1,202
ORCC fragments.

Output (committed; small): v_1/src/viz/stress_coords.json
  {"fragment_ids": [...], "embeddings":
      {"<cleaning>__<method>__L<NN>__pca": [[x,y]...], ...__tsne: ...}}
Off-cluster merge: v_1/src/viz/05_merge_stress_coords.py.

Usage:  python v_1/src/stress_tests/eda/dump_stress_coords.py
"""
from __future__ import annotations

import csv
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
OUT = _REPO / "v_1/src/viz/stress_coords.json"
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "mlm", "random"]
CLEANINGS = ["maximal", "engtier0", "engmaximal"]


def best_layer_maximal(method):
    fp = ST / "results/csv/p1_year_mc.csv"
    for r in csv.DictReader(open(fp)):
        if r["model"] == method and r["site"] == "mean_maximal" and r["best_layer"]:
            return int(r["best_layer"])
    return 0


def best_layer_translation(method, cleaning):
    fp = ST / f"translation/results/trans_mc__{method}.json"
    if not fp.exists():
        return 0
    d = json.loads(fp.read_text()).get("cleanings", {}).get(cleaning, {})
    best, best_l = -9.0, 0
    for L, blk in d.get("per_layer", {}).items():
        yr = blk.get("year", {})
        k = str(yr.get("best_k", ""))
        sp = yr.get("per_k", {}).get(k, {}).get("spearman_mean")
        if sp is not None and sp == sp and sp > best:
            best, best_l = sp, int(L)
    return best_l


def coords2d(X):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Xz = np.nan_to_num(X.astype(np.float64))
    p = PCA(n_components=2, random_state=42).fit_transform(Xz)
    t = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(Xz)
    return (np.round(p, 4).tolist(), np.round(t, 4).tolist())


def main():
    df = pd.read_parquet(CORPUS)
    out = {"fragment_ids": df["fragment_id"].astype(str).tolist(), "embeddings": {}}
    if OUT.exists():  # resumable / merge-on-write
        prev = json.loads(OUT.read_text())
        if prev.get("fragment_ids") == out["fragment_ids"]:
            out["embeddings"] = prev.get("embeddings", {})
    for cl in CLEANINGS:
        for m in MODELS:
            d = find_acts_dir(m, cl, "mean")
            if d is None:
                print(f"skip {m} x {cl} (no acts)")
                continue
            bl = (best_layer_maximal(m) if cl == "maximal"
                  else best_layer_translation(m, cl))
            layers = sorted({0, bl} & set(available_layers(d)))
            for L in layers:
                key = f"{cl}__{m}__L{L:02d}"
                if f"{key}__pca" in out["embeddings"]:
                    print(f"{key}: already dumped, skip")
                    continue
                pca, tsne = coords2d(load_layer(d, L))
                out["embeddings"][f"{key}__pca"] = pca
                out["embeddings"][f"{key}__tsne"] = tsne
                print(f"{m} x {cl} L{L}: done", flush=True)
            OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, "
          f"{len(out['embeddings'])} keys)")


if __name__ == "__main__":
    main()
