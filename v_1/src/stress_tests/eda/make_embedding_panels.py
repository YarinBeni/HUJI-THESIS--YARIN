"""make_embedding_panels.py — six-view metadata panels of the stress-test
embeddings, organized as embedding_panels/<cleaning>/<reduction>/<model>.png.

For every (cleaning in {maximal, engtier0}) x (reduction in {tsne, pca, umap,
pls}) x model, render one figure with six views of the same 2-D map: year,
ruler, period, sub-genre, provenance, fragment-length. Coordinate sources:
  tsne/pca  <- viz/stress_coords.json (best non-L0 layer) + tfidf from seal_viz_data
  umap      <- viz/stress_umap_coords.json
  pls       <- viz/pls3d_coords.json (PLS comp1 vs comp2) + tfidf computed here
Run from the repo root. Regenerates the whole tree idempotently.

Usage:  python v_1/src/stress_tests/eda/make_embedding_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
VIZ = REPO / "v_1/src/viz"
CORPUS = REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = REPO / "v_1/src/stress_tests/e6_clusters/embedding_panels"
CLEANINGS = ["maximal", "engtier0"]
df = pd.read_parquet(CORPUS)
FIDS = df["fragment_id"].astype(str).tolist()


def orcc_rows(data):
    """Return (index-into-embedding, aligned to df order) for a coords blob whose
    fragment_ids == FIDS, or the seal_viz_data layout (ORCC subset)."""
    if data.get("fragment_ids") == FIDS:
        return list(range(len(FIDS)))
    frs = data["fragments"]
    return [i for i, f in enumerate(frs) if f.get("corpus") == "orcc"]


def best_key(emб, cl, model, proj):
    cands = [k for k in emб if k.startswith(f"{cl}__{model}__") and k.endswith(f"__{proj}")]
    if not cands:
        return None
    return sorted(cands, key=lambda k: k.split("__")[2])[-1]  # highest layer (best>L0)


def coords_2d(vec):
    a = np.array([[np.nan, np.nan] if (v is None or v[0] is None) else v[:2]
                  for v in vec], dtype=float)
    return a


def panel(Z, sub, out):
    y = sub["year"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    vmin, vmax = np.nanpercentile(y, [2, 98])
    s = axes[0][0].scatter(Z[:, 0], Z[:, 1], c=np.clip(y, vmin, vmax), cmap="plasma",
                           s=11, alpha=0.85, linewidths=0)
    axes[0][0].set_title("YEAR BCE", fontsize=11)
    cb = fig.colorbar(s, ax=axes[0][0], shrink=0.8); cb.ax.invert_yaxis()

    def cat(ax, col, title, topn=8):
        vals = sub[col].astype(str); top = vals.value_counts().head(topn).index.tolist()
        oth = ~vals.isin(top)
        ax.scatter(Z[oth.values, 0], Z[oth.values, 1], c="#dcdcdc", s=6, alpha=0.5, linewidths=0)
        cmap = plt.get_cmap("tab10")
        for i, v in enumerate(top):
            m = (vals == v).values; lbl = v if len(v) <= 22 else v[:20] + "…"
            ax.scatter(Z[m, 0], Z[m, 1], color=cmap(i), s=12, alpha=0.9, linewidths=0,
                       label=f"{lbl} ({m.sum()})")
        ax.set_title(title, fontsize=11); ax.legend(fontsize=6, loc="best", framealpha=0.9)

    cat(axes[0][1], "ruler", "RULER (top 8)")
    cat(axes[0][2], "period", "PERIOD")
    cat(axes[1][0], "sub_genre", "SUB-GENRE (top 8)")
    cat(axes[1][1], "provenance", "PROVENANCE (top 8)")
    lg = np.log10(sub["word_count"].to_numpy(dtype=float).clip(1))
    s2 = axes[1][2].scatter(Z[:, 0], Z[:, 1], c=lg, cmap="viridis", s=11, alpha=0.85, linewidths=0)
    axes[1][2].set_title("LENGTH (log10 words)", fontsize=11); fig.colorbar(s2, ax=axes[1][2], shrink=0.8)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90); plt.close(fig)


def render_from(data, proj, cl, title_fmt):
    rows = orcc_rows(data)
    made = 0
    models = sorted({k.split("__")[1] for k in data["embeddings"]
                     if k.startswith(f"{cl}__") and k.endswith(f"__{proj}")})
    for m in models:
        key = best_key(data["embeddings"], cl, m, proj)
        if not key:
            continue
        vec = [data["embeddings"][key][i] for i in rows]
        Z = coords_2d(vec)
        mask = np.isfinite(Z).all(1)
        sub = df[mask].reset_index(drop=True)
        panel(Z[mask], sub, OUT / cl / proj / f"{m}.png")
        made += 1
    return made


def tfidf_pls(cl):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.cross_decomposition import PLSRegression
    tr = pd.read_parquet(REPO / "v_1/src/stress_tests/translation/translations.parquet").set_index("fragment_id")
    texts = (df["text_maximal"].fillna("").astype(str).tolist() if cl == "maximal"
             else tr["eng_tier0"].reindex(FIDS).fillna("").astype(str).tolist())
    X = TruncatedSVD(512, random_state=0).fit_transform(
        normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit_transform(texts)))
    y = df["year"].to_numpy(dtype=float); ok = np.isfinite(y)
    Z = np.full((len(df), 2), np.nan)
    Z[ok] = PLSRegression(3).fit(X[ok], y[ok]).transform(X[ok])[:, :2]
    mask = np.isfinite(Z).all(1)
    panel(Z[mask], df[mask].reset_index(drop=True), OUT / cl / "pls" / "tfidf.png")


def main():
    sc = json.loads((VIZ / "stress_coords.json").read_text())
    um = json.loads((VIZ / "stress_umap_coords.json").read_text())
    pls = json.loads((VIZ / "pls3d_coords.json").read_text())
    # rename pls3d proj key so best_key/render sees "pls"
    pls = {"fragment_ids": pls["fragment_ids"],
           "embeddings": {k.replace("__pls3d", "__pls"): v for k, v in pls["embeddings"].items()}}
    gui = json.loads((VIZ / "seal_viz_data.json").read_text())

    total = 0
    for cl in CLEANINGS:
        total += render_from(sc, "tsne", cl, None)
        total += render_from(sc, "pca", cl, None)
        total += render_from(um, "umap", cl, None)
        total += render_from(pls, "pls", cl, None)
        tfidf_pls(cl)  # tfidf pls (both cleanings, computed here)
        total += 1
        # tfidf tsne/pca from the GUI data (maximal only exists as tfidf__maximal__na__*)
        for proj in ("tsne", "pca"):
            key = f"tfidf__{cl}__na__{proj}"
            if key in gui["embeddings"]:
                rows = orcc_rows(gui)
                Z = coords_2d([gui["embeddings"][key][i] for i in rows])
                mask = np.isfinite(Z).all(1)
                panel(Z[mask], df[mask].reset_index(drop=True), OUT / cl / proj / "tfidf.png")
                total += 1
    print(f"wrote {total} panels under {OUT}")


if __name__ == "__main__":
    main()
