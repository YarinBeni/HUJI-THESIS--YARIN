"""dump_pls3d_coords.py — SUPERVISED PLS k=3 3-D coordinates of the stress-test
embeddings, for 3-D year-separability maps (k=3 was the probes' most common
best-k). CAVEAT baked into the key name: the projection is FIT ON YEAR over the
full corpus, so separation is partly supervised — compare models against the
TF-IDF and random rows under the SAME projection, not against unsupervised maps.

Per model x cleaning {maximal, engtier0} at the model's best year layer:
l2-normalize rows, fit PLSRegression(n_components=3) on the finite-year
fragments, transform all -> 3-D scores.

Output (committed; small): v_1/src/viz/pls3d_coords.json
  {"fragment_ids": [...], "embeddings": {"<cleaning>__<method>__L<NN>__pls3d":
      [[x,y,z] or [null,null,null] ...] }}

Usage:  python v_1/src/stress_tests/eda/dump_pls3d_coords.py
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
from geo_loader import find_acts_dir, load_layer, available_layers  # noqa: E402
from dump_stress_coords import best_layer_maximal, best_layer_translation  # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = _REPO / "v_1/src/viz/pls3d_coords.json"
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "mlm", "random"]
CLEANINGS = ["maximal", "engtier0"]


def main():
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import normalize

    df = pd.read_parquet(CORPUS)
    year = df["year"].to_numpy(dtype=float)
    ok = np.isfinite(year)
    out = {"fragment_ids": df["fragment_id"].astype(str).tolist(), "embeddings": {}}
    if OUT.exists():
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
            if bl not in available_layers(d):
                bl = 0
            key = f"{cl}__{m}__L{bl:02d}__pls3d"
            if key in out["embeddings"]:
                print(f"{key}: already dumped, skip")
                continue
            X = normalize(np.nan_to_num(load_layer(d, bl).astype(np.float64)))
            Z = np.full((len(df), 3), np.nan)
            pls = PLSRegression(n_components=3).fit(X[ok], year[ok])
            Z[:, :] = pls.transform(X)
            out["embeddings"][key] = [
                [None, None, None] if not np.isfinite(r).all()
                else [round(float(r[0]), 4), round(float(r[1]), 4), round(float(r[2]), 4)]
                for r in Z]
            print(f"{key}: done", flush=True)
            OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(out['embeddings'])} keys)")


if __name__ == "__main__":
    main()
