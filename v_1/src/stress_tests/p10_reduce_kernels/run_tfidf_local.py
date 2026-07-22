"""P10 TF-IDF floor (CPU, local) — reduce-then-{gkpls/rbfkpls/krr/dial} on TF-IDF.

char_wb(2,5) -> SVD-512 (the suite's TF-IDF convention), then the same reducer×norm
sweep + balanced-MC as run_acts.py. The floor the reduce-then-kernel story is read
against.

Usage:  python run_tfidf_local.py [--n-draws 100] [--n-jobs 8]
Writes  results/p10__tfidf.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
import reduce_kernels as RK  # noqa: E402

PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"


def features(df, cleaning):
    col = {"maximal": "text_maximal", "engtier0": "text_maximal"}[cleaning]
    txt = df[col].fillna("").astype(str).to_numpy()
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    return TruncatedSVD(n_components=512, random_state=42).fit_transform(V.fit_transform(txt))


def run(args):
    df = pd.read_parquet(PARQUET)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    draw_rows = [np.where(r)[0] for r in
                 np.load(BAL / "draws_matrix.npy")[: args.n_draws]]
    configs = [(red, norm) for red in RK.REDUCERS for norm in RK.NORMS]

    out = {"method": "tfidf", "protocol": "p10_reduce_kernels",
           "dims": args.dims, "n_draws": args.n_draws, "cleanings": {}}
    for cl in args.cleanings.split(","):
        X = features(df, cl)
        print(f"[tfidf x {cl}] X={X.shape}", flush=True)
        from joblib import Parallel, delayed

        def one(cfg):
            red, norm = cfg
            t0 = time.time()
            r = RK.mc_reduced(X, year, ruler, draw_rows, reducer=red, norm=norm,
                              dims=args.dims, do_gkpls=True, do_dial=True)
            print(f"    {red}/{norm}: gkpls="
                  f"{r.get('gkpls', {}).get('spearman_mean', float('nan')):.3f}"
                  f" ({time.time()-t0:.0f}s)", flush=True)
            return f"{red}/{norm}", r

        res = dict(Parallel(n_jobs=args.n_jobs)(delayed(one)(c) for c in configs))
        out["cleanings"][cl] = {"layer": None, "configs": res}

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "p10__tfidf.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {outdir / 'p10__tfidf.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cleanings", default="maximal")
    p.add_argument("--dims", type=int, default=3)
    p.add_argument("--n-draws", type=int, default=60)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", default=str(_THIS.parent / "results"))
    run(p.parse_args())
