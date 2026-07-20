"""dump_stress_umap.py — UMAP-2D coords of the stress-test embeddings, the
UMAP counterpart to dump_stress_coords.py's t-SNE/PCA. Per model x cleaning
{maximal, engtier0} at L0 + best year layer; l2-normalized rows -> UMAP-2D
(n_neighbors=30, min_dist=0.1, seed 42; cosine metric).

Output (committed; small): v_1/src/viz/stress_umap_coords.json
  {"fragment_ids": [...], "embeddings": {"<cleaning>__<method>__L<NN>__umap": [[x,y]...]}}

Usage:  python v_1/src/stress_tests/eda/dump_stress_umap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers          # noqa: E402
from dump_stress_coords import best_layer_maximal, best_layer_translation  # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = _REPO / "v_1/src/viz/stress_umap_coords.json"
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "mlm", "random"]
CLEANINGS = ["maximal", "engtier0"]


def main():
    import umap
    from sklearn.preprocessing import normalize

    df = pd.read_parquet(CORPUS)
    out = {"fragment_ids": df["fragment_id"].astype(str).tolist(), "embeddings": {}}
    if OUT.exists():
        prev = json.loads(OUT.read_text())
        if prev.get("fragment_ids") == out["fragment_ids"]:
            out["embeddings"] = prev.get("embeddings", {})
    for cl in CLEANINGS:
        for m in MODELS:
            d = find_acts_dir(m, cl, "mean")
            if d is None:
                print(f"skip {m} x {cl} (no acts)"); continue
            bl = (best_layer_maximal(m) if cl == "maximal"
                  else best_layer_translation(m, cl))
            layers = sorted({0, bl} & set(available_layers(d)))
            for L in layers:
                key = f"{cl}__{m}__L{L:02d}__umap"
                if f"{key}" in out["embeddings"]:
                    print(f"{key}: exists, skip"); continue
                X = normalize(np.nan_to_num(load_layer(d, L).astype(np.float64)))
                Z = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                              random_state=42).fit_transform(X)
                out["embeddings"][key] = np.round(Z, 3).tolist()
                print(f"{m} x {cl} L{L}: done", flush=True)
            OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(out['embeddings'])} keys)")


if __name__ == "__main__":
    main()
