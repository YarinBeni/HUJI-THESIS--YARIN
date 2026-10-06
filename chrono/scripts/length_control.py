"""Model-level length control for the language-alignment figure (advisor).

The wave-1 control was label-level: quantile bins of character length can
"align" across languages at ARI ~.65, but the models' real clusters ignore
length (ARI ~0). The advisor's stronger version: feed the MODEL texts whose
only remaining information is length — every non-space character replaced
by 'a', word boundaries kept — and measure how much cross-language
alignment the model itself produces from that.

  --build    write chrono/artifacts_tier0/lenctl.parquet
             (id = "lenctl::{doc_id}::{lang}", text = the a-text)
  --analyze  per model at its Fig-B peak layer:
               ARI(fake-akk, fake-eng) + shuffle null     <- the headline
               ARI(fake, real) per language               <- overlap check
             plus the label-level bin sweep at q = 4 / 8 / 16.
             Writes chrono/reports/tier0/atlas/CONTROLS4.md.

Extraction between the two steps is the ordinary cache pass:
  extract_embeddings.py --model <key> --table lenctl.parquet \
      --id-col id --text-col text --layers <peak> --sites mean
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.models.store import EmbStore                        # noqa: E402

ART = "chrono/artifacts_tier0"
ATLAS = "chrono/reports/tier0/atlas"
LENCTL = os.path.join(ART, "lenctl.parquet")


def a_text(s: str) -> str:
    return re.sub(r"\S", "a", str(s))


def clean_views():
    views = pd.read_parquet(os.path.join(ART, "views.parquet"))
    return views[views["augs"].fillna("") == ""].drop_duplicates(["doc_id", "lang"])


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
    a, b = np.asarray(a), np.asarray(b)
    m = (a != -1) & (b != -1)
    return adjusted_rand_score(a[m], b[m]) if m.sum() > 100 else np.nan


def build():
    clean = clean_views()
    if "text" not in clean:
        raise SystemExit("views.parquet has no text column; cannot build")
    out = pd.DataFrame({
        "id": "lenctl::" + clean.doc_id.astype(str) + "::" + clean.lang,
        "text": clean.text.map(a_text)})
    out.to_parquet(LENCTL)
    print(f"[build] {LENCTL}: {len(out)} rows "
          f"({(clean.lang == 'akk').sum()} akk, {(clean.lang == 'eng').sum()} eng)")


def analyze(seed=0):
    rng = np.random.default_rng(seed)
    store = EmbStore(os.path.join(ART, "emb_store"))
    clean = clean_views()
    B = pd.read_csv(os.path.join(ATLAS, "B_language_alignment.csv"))
    peaks = B.loc[B.groupby("model").lang_ari.idxmax(), ["model", "layer", "lang_ari"]]
    lines = ["# Length control, model-level — 'aaaa' texts through the encoder", "",
             "Every non-space character replaced by 'a'; only length and word "
             "boundaries survive. Clustered at each model's Fig-B peak layer.", "",
             "| model | layer | ARI real (orig) | ARI on a-texts | shuffle null | "
             "ARI(a, real) akk | eng |", "|---|---|---|---|---|---|---|"]
    for _, row in peaks.iterrows():
        model, L = row.model, int(row.layer)
        X, lab = {}, {}
        ok = True
        for kind, idfmt in (("real", "{d}::{l}"), ("fake", "lenctl::{d}::{l}")):
            for lang in ("akk", "eng"):
                sub = clean[clean.lang == lang]
                ids = [idfmt.format(d=d, l=lang) if kind == "fake" else v
                       for d, v in zip(sub.doc_id, sub.view_id)]
                h = np.asarray(store.has(model, L, "mean", ids), bool)
                if h.mean() < 0.8:
                    print(f"[analyze] {model} L{L} {kind}/{lang}: only "
                          f"{int(h.sum())}/{len(h)} in store — skipping model")
                    ok = False
                    break
                X[(kind, lang)] = (store.get(model, L, "mean",
                                             list(np.asarray(ids)[h])).astype(np.float32),
                                   sub.doc_id.to_numpy()[h])
            if not ok:
                break
        if not ok:
            continue
        res = {}
        for kind in ("real", "fake"):
            (Xa, da), (Xe, de) = X[(kind, "akk")], X[(kind, "eng")]
            common, ia, ie = np.intersect1d(da, de, return_indices=True)
            la, le = clusters(Xa[ia], seed), clusters(Xe[ie], seed)
            lab[(kind, "akk")] = pd.Series(la, index=common)
            lab[(kind, "eng")] = pd.Series(le, index=common)
            res[kind] = ari(la, le)
        null = float(np.nanmean([ari(lab[("fake", "akk")].to_numpy(),
                                     rng.permutation(lab[("fake", "eng")].to_numpy()))
                                 for _ in range(10)]))
        overlap = {}
        for lang in ("akk", "eng"):
            a, b = lab[("fake", lang)], lab[("real", lang)]
            ix = a.index.intersection(b.index)
            overlap[lang] = ari(a.loc[ix].to_numpy(), b.loc[ix].to_numpy())
        lines.append(f"| `{model.split('/')[-1]}` | {L} | {res['real']:.3f} | "
                     f"{res['fake']:.3f} | {null:+.3f} | {overlap['akk']:.3f} | "
                     f"{overlap['eng']:.3f} |")
        print(lines[-1], flush=True)

    # label-level bin sweep: is q=8 doing the work?
    lines += ["", "## Label-level control, bin sweep", "",
              "ARI between akk and eng length-quantile bins of the same documents.",
              "", "| q bins | ARI |", "|---|---|"]
    wide = clean.pivot_table(index="doc_id", columns="lang", values="text",
                             aggfunc="first").dropna()
    for q in (4, 8, 16):
        bins = {l: pd.qcut(wide[l].str.len(), q=q, labels=False,
                           duplicates="drop").to_numpy() for l in ("akk", "eng")}
        lines.append(f"| {q} | {ari(bins['akk'], bins['eng']):.3f} |")
        print(lines[-1], flush=True)

    open(os.path.join(ATLAS, "CONTROLS4.md"), "w").write("\n".join(lines) + "\n")
    print("wrote CONTROLS4.md")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.build:
        build()
    if args.analyze:
        analyze(args.seed)


if __name__ == "__main__":
    main()
