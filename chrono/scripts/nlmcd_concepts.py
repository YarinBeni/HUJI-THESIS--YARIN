"""P6.4 via NLMCD (arXiv:2610.01821, adapted): concepts as manifolds, not axes.

For each chosen representation: UMAP the document embeddings to a few
dimensions, cluster with HDBSCAN, and treat each cluster as a candidate
"concept manifold". Then ask the two questions that matter for the thesis:

1. WHAT is each manifold? Label every cluster by the majority source /
   period / genre and its purity. (Prediction from everything so far: the
   manifolds are corpora.)
2. Is there TIME inside them? Within each cluster that is dominated by one
   source, measure period sub-structure (silhouette of period labels in the
   full-dimensional space, against a permutation null) — time-as-a-manifold
   would show up here even if no global time axis exists.

Across representations we report an adjusted-Rand alignment matrix on the
shared documents (noise points excluded) — the spirit of the paper's CBA
score (their generalized Rand index), not a reimplementation of it.

    python nlmcd_concepts.py            # all default cells present in the store
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.models.store import EmbStore                        # noqa: E402

CELLS = [  # (model, layer, site) — frozen, SSL-adapter, from-scratch, hybrid
    ("Thalesian/cuneiformBase-400m", 12, "mean"),
    ("Qwen/Qwen3-8B", 27, "mean"),
    ("ssl::ssl_byol_cunei400m_wdated-s0", 0, "h"),
    ("ssl_e2e::e2e_barlow_S-s0", 0, "h"),
]


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def silhouette_null(X, y, seed, n_perm=100):
    from sklearn.metrics import silhouette_score
    s = silhouette_score(X, y)
    rng = np.random.default_rng(seed)
    null = [silhouette_score(X, rng.permutation(y)) for _ in range(n_perm)]
    return s, float(np.mean(null)), float(np.std(null)), float(np.mean([n >= s for n in null]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="chrono/artifacts_ssl/corpus_all.parquet")
    ap.add_argument("--store-root", default="chrono/artifacts_ssl/emb_store")
    ap.add_argument("--out", default="chrono/reports/ssl/NLMCD_RESULT.md")
    ap.add_argument("--umap-dim", type=int, default=8)
    ap.add_argument("--min-cluster", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import umap
    try:
        from sklearn.cluster import HDBSCAN as _HDB            # sklearn >= 1.3
        hdb = lambda: _HDB(min_cluster_size=args.min_cluster)
    except ImportError:                                         # fallback package
        import hdbscan
        hdb = lambda: hdbscan.HDBSCAN(min_cluster_size=args.min_cluster)

    c = pd.read_parquet(args.corpus)
    c = c[c["split"] != "dated"].reset_index(drop=True)
    ids = ("ssl::" + c.uid.astype(str)).tolist()
    store = EmbStore(args.store_root)

    labels_by_cell, lines = {}, ["# NLMCD-style concept manifolds", "",
        f"UMAP->{args.umap_dim}d, HDBSCAN(min_cluster_size={args.min_cluster}) over "
        f"{len(c):,} undated documents; clusters labelled by majority metadata. "
        "Within source-dominated clusters: period silhouette (full-dim) vs a "
        "100-permutation null. Cross-representation alignment: adjusted Rand on "
        "shared non-noise documents (CBA-inspired).", ""]
    for model, layer, site in CELLS:
        try:
            if not np.all(store.has(model, layer, site, ids)):
                lines += [f"## `{model}::L{layer}::{site}` — SKIPPED (embeddings missing)", ""]
                continue
            X = store.get(model, layer, site, ids).astype(np.float32)
        except Exception as e:                                  # noqa: BLE001
            lines += [f"## `{model}::L{layer}::{site}` — SKIPPED ({type(e).__name__})", ""]
            continue
        U = umap.UMAP(n_components=args.umap_dim, random_state=args.seed).fit_transform(X)
        lab = hdb().fit(U).labels_
        labels_by_cell[f"{model}::L{layer}::{site}"] = lab
        ks = sorted(set(lab) - {-1})
        lines += [f"## `{model}::L{layer}::{site}` — {len(ks)} manifolds, "
                  f"{int((lab == -1).sum()):,} noise docs", "",
                  "| cluster | n | majority source (purity) | majority period (purity) "
                  "| majority genre (purity) | period silhouette inside | p |",
                  "|---|---|---|---|---|---|---|"]
        for k in ks:
            m = lab == k
            row = []
            for col in ("source", "period_norm", "genre_raw"):
                v = c.loc[m, col].dropna()
                row.append(f"{v.mode().iat[0]} ({(v == v.mode().iat[0]).mean():.2f})"
                           if len(v) else "—")
            sil = "—"; pv = "—"
            v = c.loc[m, "period_norm"].dropna()
            vc = v.value_counts()
            keep = vc[vc >= 20].index
            if len(keep) >= 2:
                mm = m & c.period_norm.isin(keep).to_numpy()
                s, mu, sd, p = silhouette_null(X[mm], c.loc[mm, "period_norm"].to_numpy(),
                                               args.seed)
                sil, pv = f"{s:+.3f} (null {mu:+.3f}±{sd:.3f})", f"{p:.2f}"
            lines.append(f"| {k} | {int(m.sum()):,} | {row[0]} | {row[1]} | {row[2]} "
                         f"| {sil} | {pv} |")
        lines.append("")

    if len(labels_by_cell) >= 2:
        from sklearn.metrics import adjusted_rand_score
        names = list(labels_by_cell)
        lines += ["## Cross-representation alignment (adjusted Rand, non-noise overlap)", "",
                  "| | " + " | ".join(f"`{n.split('::')[0].split('/')[-1]}`" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for a in names:
            cells = []
            for b in names:
                la, lb = labels_by_cell[a], labels_by_cell[b]
                m = (la != -1) & (lb != -1)
                cells.append(f"{adjusted_rand_score(la[m], lb[m]):.3f}" if m.sum() > 100 else "—")
            lines.append(f"| `{a.split('::')[0].split('/')[-1]}` | " + " | ".join(cells) + " |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    # keep assignments for later drill-down
    out_pq = os.path.join(os.path.dirname(args.out), "nlmcd_clusters.parquet")
    pd.DataFrame({"uid": c.uid, **{_safe(k): v for k, v in labels_by_cell.items()}}
                 ).to_parquet(out_pq, index=False)
    print(f"wrote {args.out} and {out_pq}")


if __name__ == "__main__":
    main()
