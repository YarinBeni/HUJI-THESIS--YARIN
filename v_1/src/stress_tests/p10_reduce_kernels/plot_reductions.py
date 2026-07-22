"""P10 visualization — 3D embeddings of the activations, before and after
normalization, colored by year and by ruler.

For a method×cleaning (at the P9-best layer) or TF-IDF, fit PCA / PLS / UMAP / t-SNE
to `dims`=3 on a subsample and draw a panel grid:
    rows = {PCA, PLS, UMAP, t-SNE}
    cols = {year (raw), ruler (raw), year (z-scored), ruler (z-scored)}
This is EXPLORATORY whole-data viz (fit on all points, no CV) — it shows whether a
low-dim view separates chronology, motivating the reduce-then-kernel probe. PLS is
supervised (uses year) so its view is the best-case projection.

Usage: python plot_reductions.py --method qwen3_8b --cleaning maximal
       python plot_reductions.py --tfidf
Writes results/fig_p10_reductions__{tag}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA                  # noqa: E402
from sklearn.manifold import TSNE                      # noqa: E402

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
PARQUET = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"


def _reduce(kind, X, y, seed=42):
    d = min(3, X.shape[1] - 1)
    if kind == "PCA":
        return PCA(n_components=d, random_state=seed).fit_transform(X)
    if kind == "PLS":
        return PLSRegression(n_components=d, scale=False).fit(X, y).transform(X)
    if kind == "UMAP":
        import umap
        return umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                         random_state=seed).fit_transform(X)
    if kind == "t-SNE":
        return TSNE(n_components=3, random_state=seed,
                    perplexity=min(30, len(X) // 4)).fit_transform(X)
    raise ValueError(kind)


def _zscore(Z):
    sd = Z.std(0)
    return (Z - Z.mean(0)) / np.where(sd == 0, 1.0, sd)


def make_figure(X, year, ruler, title, out_path, seed=42):
    reducers = ["PCA", "PLS", "UMAP", "t-SNE"]
    cols = [("year", False), ("ruler", False), ("year", True), ("ruler", True)]
    rcodes = pd.Series(ruler).astype("category").cat.codes.to_numpy()
    fig = plt.figure(figsize=(4.3 * len(cols), 4.0 * len(reducers)))
    for ri, red in enumerate(reducers):
        try:
            Z = np.asarray(_reduce(red, X, year, seed))
        except Exception as e:  # noqa: BLE001
            print(f"[{red}] failed: {type(e).__name__}: {e}")
            continue
        for ci, (color_by, zn) in enumerate(cols):
            ax = fig.add_subplot(len(reducers), len(cols),
                                 ri * len(cols) + ci + 1, projection="3d")
            Zp = _zscore(Z) if zn else Z
            c = year if color_by == "year" else rcodes
            cmap = "viridis" if color_by == "year" else "tab20"
            p = ax.scatter(Zp[:, 0], Zp[:, 1], Zp[:, 2] if Zp.shape[1] > 2 else 0,
                           c=c, cmap=cmap, s=8, alpha=0.7)
            ax.set_title(f"{red} · {color_by}{' · z' if zn else ''}", fontsize=9)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if color_by == "year":
                fig.colorbar(p, ax=ax, shrink=0.5, pad=0.02)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def _subsample(X, year, ruler, n, seed=42):
    if len(X) <= n:
        return X, year, ruler
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], year[idx], ruler[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method")
    ap.add_argument("--cleaning", default="maximal")
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    keep = np.isfinite(year)
    outdir = _THIS.parent / "results"

    if args.tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        txt = df["text_maximal"].fillna("").astype(str).to_numpy()[keep]
        V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
        X = TruncatedSVD(n_components=512, random_state=42).fit_transform(V.fit_transform(txt))
        y2, r2 = year[keep], ruler[keep]
        X, y2, r2 = _subsample(X, y2, r2, args.n)
        make_figure(X, y2, r2, "TF-IDF (char 2-5 → SVD-512) — maximal",
                    outdir / "fig_p10_reductions__tfidf.png")
        return

    sys.path.insert(0, str(_THIS.parent))
    from geo_loader import find_acts_dir, load_layer, available_layers
    import json
    d = find_acts_dir(args.method, args.cleaning, "mean")
    if d is None:
        print(f"acts missing for {args.method} x {args.cleaning}"); return
    layers = available_layers(d)
    p9 = _THIS.parents[1] / "p9_gkpls" / "results" / f"p9_gkpls__{args.method}.json"
    L = layers[len(layers) // 2]
    if p9.exists():
        bl = json.loads(p9.read_text()).get("cleanings", {}).get(args.cleaning, {}).get("best_layer")
        if isinstance(bl, int) and bl in layers:
            L = bl
    X = load_layer(d, L)[keep]
    y2, r2, Xs = year[keep], ruler[keep], X
    Xs, y2, r2 = _subsample(Xs, y2, r2, args.n)
    make_figure(Xs, y2, r2, f"{args.method} · {args.cleaning} · layer {L}",
                outdir / f"fig_p10_reductions__{args.method}__{args.cleaning}.png")


if __name__ == "__main__":
    main()
