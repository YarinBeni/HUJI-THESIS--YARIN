"""P9 local runner — G-KPLS / RBF-KPLS / KRR on TF-IDF features (CPU).

TF-IDF control row for the P9 slide: char_wb(2,5) -> SVD-512 (the suite's
baseline convention), same balanced-MC protocol as the model runs, on the
canonical pair maximal (Akkadian, name-stripped) + engtier0 (English, the
only valid translation — eng_maximal made the translator hallucinate names).

Usage:  python run_tfidf_local.py [--cleanings maximal,engtier0]
Writes  results/p9_gkpls__tfidf.json (merged if it exists)
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
from gkpls import mc_gkpls_probe  # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"
ARMS = ["gkpls", "rbfkpls", "krr_geo"]
SVD_DIM = 512


def texts_for(df, cleaning):
    if cleaning in ("tier0", "maximal"):
        return df[f"text_{cleaning}"].fillna("").astype(str).tolist()
    col = "eng_tier0" if cleaning == "engtier0" else "eng_maximal"
    tr = pd.read_parquet(TRANSLATIONS).set_index("fragment_id")[col]
    return tr.reindex(df["fragment_id"].astype(str)).fillna("").astype(str).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleanings", default="maximal,engtier0")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    a = p.parse_args()

    df = pd.read_parquet(PARQUET)
    fids = json.load(open(BAL / "corpus_fragment_order.json"))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    dm = np.load(BAL / "draws_matrix.npy")[: a.n_draws]
    draw_rows = [np.where(r)[0] for r in dm]

    out = {"method": "tfidf", "protocol": "p9_gkpls_mc", "k_neighbors": a.k,
           "tfidf": "char_wb(2,5), SVD-512 (baseline convention)", "cleanings": {}}
    for cl in a.cleanings.split(","):
        t0 = time.time()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        Xs = normalize(vec.fit_transform(texts_for(df, cl)), norm="l2")
        svd = TruncatedSVD(n_components=min(SVD_DIM, Xs.shape[1] - 1), random_state=0)
        X = svd.fit_transform(Xs).astype(np.float32)
        r = mc_gkpls_probe(X, year, ruler, draw_rows, k=a.k)
        blk = {"per_layer": {"0": r}, "best_layer": 0,
               "best": {arm: r[arm] for arm in ARMS}}
        out["cleanings"][cl] = blk
        b = blk["best"]
        print(f"[{cl}] gkpls={b['gkpls']['spearman_mean']:.3f}(a={b['gkpls']['best_hyper']})"
              f"  rbfkpls={b['rbfkpls']['spearman_mean']:.3f}(a={b['rbfkpls']['best_hyper']})"
              f"  krr_geo={b['krr_geo']['spearman_mean']:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "p9_gkpls__tfidf.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8")).get("cleanings", {})
        out["cleanings"] = {**prev, **out["cleanings"]}
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
