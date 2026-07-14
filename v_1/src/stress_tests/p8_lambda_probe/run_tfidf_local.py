"""P8 local runner — the lambda-dial probe on TF-IDF features (CPU, minutes).

SETUP   TF-IDF char_wb(2,5) (the suite's cited baseline config), LSA/SVD to
        512 dims (geometry-preserving densification), tier0 + maximal
        cleanings; 200 balanced draws x GroupKFold-by-ruler (the standard MC).
PROBE   lambda-probe (see MATH_NOTES.md): lambda grid 0..1, d=3,
        k in {5,10,20} neighbors.
METRIC  align1 = |Spearman(leading coord, year)| on held-out rulers;
        pred = Spearman of ridge-on-Z_d predictions.

Usage:  python run_tfidf_local.py [--n-draws 200] [--ks 5,10,20]
Writes  results/p8_lambda__tfidf.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
from lambda_probe import mc_lambda_probe, LAMBDAS  # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
CLEANINGS = ["tier0", "maximal"]
TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"
SVD_DIM = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--ks", default="5,10,20")
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    p.add_argument("--cleanings", default="", help="override, e.g. engtier0")
    a = p.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    df = pd.read_parquet(PARQUET)
    fids = json.load(open(BAL / "corpus_fragment_order.json"))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    dm = np.load(BAL / "draws_matrix.npy")[: a.n_draws]
    draw_rows = [np.where(r)[0] for r in dm]
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()

    out = {"method": "tfidf", "protocol": "p8_lambda_mc",
           "tfidf": "char_wb(2,5) fit on full corpus (baseline convention)",
           "svd_dim": SVD_DIM, "d": a.d, "lambdas": LAMBDAS, "cleanings": {}}
    for cl in (a.cleanings.split(",") if a.cleanings else CLEANINGS):
        if cl in ("tier0", "maximal"):
            texts = df[f"text_{cl}"].fillna("").astype(str).tolist()
        else:
            col = "eng_tier0" if cl == "engtier0" else "eng_maximal"
            tr = pd.read_parquet(TRANSLATIONS).set_index("fragment_id")[col]
            texts = tr.reindex(df["fragment_id"].astype(str)).fillna("").astype(str).tolist()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        Xs = normalize(vec.fit_transform(texts), norm="l2")
        svd = TruncatedSVD(n_components=min(SVD_DIM, Xs.shape[1] - 1),
                           random_state=0)
        X = svd.fit_transform(Xs).astype(np.float32)
        print(f"[{cl}] tfidf {Xs.shape} -> svd {X.shape} "
              f"(evr {svd.explained_variance_ratio_.sum():.3f})", flush=True)
        blk = {}
        for k in ks:
            t0 = time.time()
            res = mc_lambda_probe(X, year, ruler, draw_rows,
                                  k=k, d=a.d, l2_normalize=True)
            blk[f"k{k}"] = res
            pl = res.get("per_lambda", {})
            line = "  ".join(f"L{lam}: {v['align1_mean']:.3f}/{v['pred_mean']:.3f}"
                             for lam, v in pl.items() if lam in ("0.0", "0.5", "1.0"))
            print(f"  k={k} ({time.time()-t0:.0f}s)  align1/pred  {line}", flush=True)
        out["cleanings"][cl] = blk

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "p8_lambda__tfidf.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8")).get("cleanings", {})
        out["cleanings"] = {**prev, **out["cleanings"]}
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
