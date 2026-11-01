"""Controls for the language-alignment figure (advisor request).

Fig B reported ARI ~.9 between the Akkadian and English manifolds of the same
documents in the LLMs' middle layers. Before the paper can claim anything,
two alternative explanations must be priced:

C1  Chance. ARI between the akkadian clustering and a SHUFFLED pairing of
    the english clustering (20 shuffles -> null mean +- sd). If the observed
    ARI is inside the null, the figure means nothing.
C2  Trivial features. Documents keep their genre and length under
    translation. Cluster each language by DOCUMENT LENGTH alone and measure
    (a) ARI(akk-length, eng-length) — how much "alignment" length alone buys;
    (b) ARI(real clusters, length clusters) per language — how much of the
    real structure is just length.

Run at each model's peak-alignment layer (from B_language_alignment.csv).
Writes chrono/reports/tier0/atlas/B_controls.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.models.store import EmbStore                        # noqa: E402


def clusters(X, seed, min_cluster=40, dim=8):
    import umap
    try:
        from sklearn.cluster import HDBSCAN as HDB
        algo = HDB(min_cluster_size=min_cluster)
    except ImportError:
        import hdbscan
        algo = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    U = umap.UMAP(n_components=dim, random_state=seed).fit_transform(X)
    return algo.fit(U).labels_


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    m = (a != -1) & (b != -1)
    return adjusted_rand_score(a[m], b[m]) if m.sum() > 100 else np.nan


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--atlas-dir", default="chrono/reports/tier0/atlas")
    ap.add_argument("--site", default="mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-shuffles", type=int, default=20)
    args = ap.parse_args(argv)

    B = pd.read_csv(os.path.join(args.atlas_dir, "B_language_alignment.csv"))
    peaks = B.loc[B.groupby("model").lang_ari.idxmax(), ["model", "layer", "lang_ari"]]

    store = EmbStore(os.path.join(args.art, "emb_store"))
    views = pd.read_parquet(os.path.join(args.art, "views.parquet"))
    clean = views[views["augs"].fillna("") == ""].drop_duplicates(["doc_id", "lang"])
    txt_len = clean.set_index(["doc_id", "lang"]).text.str.len() if "text" in clean \
        else None

    rng = np.random.default_rng(args.seed)
    lines = ["# Fig B controls — chance and trivial-feature prices", "",
             "| model | layer | observed ARI | shuffle null (mean±sd) | "
             "length-only ARI | ARI(real, length) akk | eng |",
             "|---|---|---|---|---|---|---|"]
    for _, row in peaks.iterrows():
        model, L = row.model, int(row.layer)
        sub = {l: clean[clean.lang == l] for l in ("akk", "eng")}
        ids = {l: np.asarray(s.view_id.tolist()) for l, s in sub.items()}
        has = {l: np.asarray(store.has(model, L, args.site, list(ids[l])), bool)
               for l in ids}
        if min(h.mean() for h in has.values()) < 0.8:
            continue
        X, docs = {}, {}
        for l in ("akk", "eng"):
            X[l] = store.get(model, L, args.site, list(ids[l][has[l]])).astype(np.float32)
            docs[l] = sub[l].doc_id.to_numpy()[has[l]]
        common, ia, ie = np.intersect1d(docs["akk"], docs["eng"], return_indices=True)
        la = clusters(X["akk"][ia], args.seed)
        le = clusters(X["eng"][ie], args.seed)
        observed = ari(la, le)
        null = [ari(la, rng.permutation(le)) for _ in range(args.n_shuffles)]
        # length-only clustering: quantile bins of character length
        if txt_len is not None:
            lens = {l: txt_len.loc[list(zip(common, [l] * len(common)))].to_numpy()
                    for l in ("akk", "eng")}
            bins = {l: pd.qcut(v, q=8, labels=False, duplicates="drop").astype(int)
                    for l, v in lens.items()}
            len_ari = ari(np.asarray(bins["akk"]), np.asarray(bins["eng"]))
            real_vs_len = {l: ari(lab, np.asarray(bins[l]))
                           for l, lab in (("akk", la), ("eng", le))}
        else:
            len_ari, real_vs_len = np.nan, {"akk": np.nan, "eng": np.nan}
        lines.append(f"| `{model.split('/')[-1]}` | {L} | {observed:.3f} | "
                     f"{np.nanmean(null):+.3f}±{np.nanstd(null):.3f} | {len_ari:.3f} | "
                     f"{real_vs_len['akk']:.3f} | {real_vs_len['eng']:.3f} |")
        print(lines[-1], flush=True)

    open(os.path.join(args.atlas_dir, "B_controls.md"), "w").write("\n".join(lines) + "\n")
    print("wrote B_controls.md")


if __name__ == "__main__":
    main()
