"""J8 — P3 timeline analysis (CPU).

Using the anchor embeddings from J5:
  3a  Do the anchors form an ordered timeline? Fit a 1-D embedding (PCA-1D linear
      and Isomap-1D nonlinear) over the anchors and measure Spearman(coord, year).
      High => the model holds an internal timeline of these dates.
  3b  Do the ORCC texts land on that timeline? For each text's mean embedding,
      take its nearest anchor (cosine) -> predicted year; Spearman(pred, true).
      High 3a + low 3b = declarative knowledge not connected to text reps
      (the dissociation, rendered geometrically).

Emits results/p3_timeline__<method>.json (per layer).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from geo_loader import find_acts_dir, load_layer, available_layers, isomap_1d  # noqa: E402
from sklearn.decomposition import PCA                                     # noqa: E402
from sklearn.preprocessing import normalize                              # noqa: E402
from scipy.stats import spearmanr                                        # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"


def _sp(a, b):
    r = spearmanr(a, b).statistic
    return float(abs(r)) if r == r else float("nan")   # abs: 1-D sign is arbitrary


def run(args):
    anchor_dir = Path(args.anchors) / args.method
    if not anchor_dir.exists():
        print(f"no anchors for {args.method}"); return
    df = pd.read_parquet(CORPUS)
    ymask = df["year"].notna().to_numpy()
    true_year = df["year"].to_numpy()[ymask]

    mean_dir = find_acts_dir(args.method, "tier0", "mean")
    out = {"method": args.method, "per_layer": {}}

    for npz in sorted(anchor_dir.glob("L*.npz")):
        L = int(npz.stem[1:])
        a = np.load(npz, allow_pickle=True)
        A = normalize(a["acts"], norm="l2"); ay = a["years"].astype(float)
        # 3a: anchors form a timeline?
        pca1 = PCA(n_components=1, random_state=42).fit_transform(A).ravel()
        try:
            iso1 = isomap_1d(A, k=min(10, len(A) - 1), metric="cosine")
        except Exception:
            iso1 = np.full(len(A), np.nan)
        rec = {"n_anchors": int(len(A)),
               "3a_pca1_spearman": _sp(pca1, ay),
               "3a_isomap1_spearman": _sp(iso1, ay)}
        # 3b: ORCC texts project onto anchors (nearest anchor -> its year)
        if mean_dir is not None and L in available_layers(mean_dir):
            X = normalize(load_layer(mean_dir, L)[ymask], norm="l2")
            sims = X @ A.T                       # cosine (both l2-normalized)
            pred_year = ay[sims.argmax(axis=1)]
            rec["3b_project_spearman"] = _sp(pred_year, true_year)
            rec["3b_n_texts"] = int(len(true_year))
        out["per_layer"][L] = rec

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p3_timeline__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # headline: best layer by 3a isomap
    pl = out["per_layer"]
    if pl:
        bestL = max(pl, key=lambda L: pl[L].get("3a_isomap1_spearman", 0) or 0)
        r = pl[bestL]
        print(f"  {args.method} L{bestL}: 3a(anchors)={r.get('3a_isomap1_spearman'):.3f} "
              f"3b(texts)={r.get('3b_project_spearman', float('nan')):.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--anchors", default=str(Path(__file__).resolve().parent / "anchors"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
