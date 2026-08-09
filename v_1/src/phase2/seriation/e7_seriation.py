"""E7 — spectral seriation: does the fragment cloud have a natural 1-D order,
and what does that order correspond to?

THE ONE LABEL-FREE EXPERIMENT. Build a kNN similarity graph over fragment
activations, take the Fiedler vector (eigenvector of the 2nd-smallest eigenvalue
of the normalized graph Laplacian) — for data with an underlying 1-D structure
this recovers the ordering (Atkins, Boman & Hendrickson 1998). NO label touches
the construction; labels are used ONCE, post hoc, to ask what the order IS:
chronology? genre? provenance? length? Whatever it matches most is what actually
organizes the representation space — either answer is informative.

Read-outs per arm (sign-invariant, since an eigenvector's sign is arbitrary):
  * |Spearman(fiedler, year)| + a ruler-level permutation null (years shuffled
    among rulers, B=1000 — no refit needed, the ordering never saw labels);
  * eta-squared of the fiedler value against genre / ruler / provenance
    (categorical variance explained), |Spearman| against log word-count.

    python e7_seriation.py --method olmo2_7b --variant akk_maximal
    python e7_seriation.py --method tfidf_char --variant akk_maximal
    python e7_seriation.py --method olmo2_7b --variant akk_maximal --ruler Esarhaddon

Writes results/{method}.{variant}[.{ruler}].json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csgraph

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
RESULTS = os.path.join(_HERE, "results")


def fiedler(X, k=10):
    """Fiedler vector of the symmetrized cosine-kNN graph, largest component."""
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(X, n_neighbors=k, metric="cosine", mode="distance")
    A.data = 1.0 - A.data                    # distance -> similarity in [0,1]
    A = A.maximum(A.T)                       # symmetrize
    ncomp, labels = csgraph.connected_components(A, directed=False)
    keep = np.arange(A.shape[0])
    if ncomp > 1:
        main = np.bincount(labels).argmax()
        keep = np.where(labels == main)[0]
        A = A[keep][:, keep]
    L = csgraph.laplacian(A, normed=True)
    # dense eigendecomposition: n <= ~1.2k so this is trivial, and it avoids
    # the shift-invert-at-zero factorization failure eigsh(sigma=0) risks on a
    # singular Laplacian
    vals, vecs = np.linalg.eigh(np.asarray(L.todense()))
    return vecs[:, np.argsort(vals)[1]], keep, ncomp


def eta_sq(x, cats):
    """Variance in x explained by a categorical variable."""
    df = pd.DataFrame({"x": x, "c": pd.Categorical(cats).codes})
    grand = df.x.mean()
    ss_b = df.groupby("c").apply(
        lambda g: len(g) * (g.x.mean() - grand) ** 2,
        include_groups=False).sum()
    ss_t = ((df.x - grand) ** 2).sum()
    return float(ss_b / max(ss_t, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal", choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--ruler", default=None,
                    help="restrict to one ruler (E6 tie-in: Esarhaddon)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    # provenance is already on the eligible frame; merge only what is missing
    meta = pd.read_parquet(os.path.join(
        os.path.dirname(_WM), "..", "data/evaluation/corpora/orcc_corpus.parquet"
    ))[["fragment_id", "genre", "word_count"]]
    df = df.merge(meta, on="fragment_id", how="left")
    if args.ruler:
        df = df[df.ruler == args.ruler].reset_index(drop=True)
    print(f"[data] {len(df)} fragments"
          + (f" ({args.ruler} only)" if args.ruler else ""), flush=True)

    # feature space at the F1-selected layer (or SVD for the floor)
    if args.method == "tfidf_char":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df[PP.TEXT_COL[args.variant]].fillna("").astype(str).values
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=120000, sublinear_tf=True)
        X = TruncatedSVD(min(256, len(df) - 1), random_state=0).fit_transform(
            v.fit_transform(texts)).astype(np.float32)
        bl = -1
    else:
        f1 = os.path.join(_PAIRS, "results", "probes",
                          f"{args.method}.{args.variant}.{args.site}.json")
        bl = json.load(open(f1))["best_layer"] if os.path.exists(f1) else None
        layers = PP.load_act_layers(args.method, args.variant, args.site,
                                    stride=1)
        if bl not in layers:
            bl = sorted(layers)[len(layers) * 2 // 3]     # a mid-late default
            print(f"[warn] no F1 layer; using {bl}", flush=True)
        X = layers[bl][df.pos.values]

    fv, keep, ncomp = fiedler(X, args.knn)
    d = df.iloc[keep]
    year = d.year.values.astype(float)
    rho_year = float(abs(stats.spearmanr(fv, year).correlation))

    # permutation null for the year correlation. Ruler-level shuffle is the
    # honest null for the full corpus (years travel with rulers); inside a
    # single-ruler subset there is only one ruler, so the shuffle happens at
    # the fragment level instead — otherwise the null is degenerate.
    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_perm)
    if args.ruler:
        for i in range(args.n_perm):
            null[i] = abs(stats.spearmanr(
                fv, rng.permutation(year)).correlation)
    else:
        ruler_year = d.groupby("ruler").year.mean()
        for i in range(args.n_perm):
            perm = pd.Series(rng.permutation(ruler_year.values),
                             index=ruler_year.index)
            null[i] = abs(stats.spearmanr(
                fv, d.ruler.map(perm).values).correlation)
    p_year = float((1 + (null >= rho_year).sum()) / (args.n_perm + 1))

    out = {
        "method": args.method, "variant": args.variant, "site": args.site,
        "layer": bl, "knn": args.knn, "ruler_subset": args.ruler,
        "n": int(len(d)), "n_components": int(ncomp),
        "rho_year_abs": rho_year, "p_year_ruler_perm": p_year,
        "null_mean": float(null.mean()),
        "eta2_genre": eta_sq(fv, d.genre.fillna("unk")),
        "eta2_ruler": eta_sq(fv, d.ruler),
        "eta2_provenance": eta_sq(fv, d.provenance.fillna("unk")),
        "rho_length_abs": float(abs(stats.spearmanr(
            fv, np.log1p(d.word_count.fillna(0))).correlation)),
    }
    print(f"[seriation] |rho_year|={rho_year:.3f} (p={p_year:.3f}, "
          f"null {null.mean():.3f}) | eta2 genre={out['eta2_genre']:.3f} "
          f"ruler={out['eta2_ruler']:.3f} prov={out['eta2_provenance']:.3f} "
          f"|rho_len|={out['rho_length_abs']:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    sfx = f".{args.ruler}" if args.ruler else ""
    pth = os.path.join(RESULTS, f"{args.method}.{args.variant}{sfx}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
