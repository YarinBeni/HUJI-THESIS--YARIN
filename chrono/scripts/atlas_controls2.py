"""Deconfounding wave for the alignment analysis (advisor, 2026-10-06).

V1  LENGTH/NAMES BY CONSTRUCTION. Rerun the language-alignment measurement
    on the stored [mask_ruler + crop32] condition views: royal names masked,
    text cropped to a 32-word window — so neither names nor length can carry
    the alignment. Compare ARI(orig) vs ARI(masked+cropped) at each model's
    peak layer.
V3  TIME AS CATEGORIES VS TIME AS GEOMETRY. (a) ARI between the document
    clusters and CENTURY bins: do the manifolds agree with time read as an
    UNORDERED label? (b) Inside each century with enough documents: 5-fold
    ridge Spearman — is there year ORDER inside the category?
V4  TARGET SENSITIVITY. The entity target was the median attested year per
    ruler, but a reign is a RANGE. Re-score the name probes with target =
    first / median / last attested year; if rho moves a lot, the target
    choice matters and the paper must say which convention it uses.

Everything runs from stored artifacts; CPU only.
Writes chrono/reports/tier0/atlas/CONTROLS2.md.
"""
from __future__ import annotations

import glob
import json
import os
import re
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
    a, b = np.asarray(a), np.asarray(b)
    m = (a != -1) & (b != -1)
    return adjusted_rand_score(a[m], b[m]) if m.sum() > 100 else np.nan


def grab(store, model, L, site, view_ids, doc_ids):
    h = np.asarray(store.has(model, int(L), site, list(view_ids)), bool)
    if h.sum() < 0.6 * len(view_ids):
        return None, None
    X = store.get(model, int(L), site, list(np.asarray(view_ids)[h])).astype(np.float32)
    return X, np.asarray(doc_ids)[h]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--atlas-dir", default="chrono/reports/tier0/atlas")
    ap.add_argument("--site", default="mean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    lines = ["# Controls wave 2 — length/names, categories-vs-order, target choice", ""]

    store = EmbStore(os.path.join(args.art, "emb_store"))
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    views = pd.read_parquet(os.path.join(args.art, "views.parquet"))
    aug = views["augs"].fillna("").astype(str)
    t_of = corpus.set_index("doc_id")["t"].astype(float)

    def pick(lang, want_crop):
        if want_crop:
            m = aug.str.contains("mask_ruler") & aug.str.contains("crop32") \
                & (views.lang == lang)
        else:
            m = (aug == "") & (views.lang == lang)
        sub = views[m].drop_duplicates("doc_id")
        return sub.view_id.to_numpy(), sub.doc_id.to_numpy()

    B = pd.read_csv(os.path.join(args.atlas_dir, "B_language_alignment.csv"))
    peaks = B.loc[B.groupby("model").lang_ari.idxmax(), ["model", "layer", "lang_ari"]]

    # ---- V1: alignment with names masked and length fixed ------------------
    lines += ["## V1 — language alignment on [mask_ruler + crop32] views", "",
              "| model | layer | ARI orig | ARI masked+cropped | shuffle null |",
              "|---|---|---|---|---|"]
    for _, row in peaks.iterrows():
        model, L = row.model, int(row.layer)
        out = {}
        for cond, crop in (("orig", False), ("mc32", True)):
            la = le = None
            got = {}
            for lang in ("akk", "eng"):
                vid, did = pick(lang, crop)
                X, docs = grab(store, model, L, args.site, vid, did)
                if X is None:
                    got = None; break
                got[lang] = (X, docs)
            if not got:
                out[cond] = (np.nan, np.nan); continue
            common, ia, ie = np.intersect1d(got["akk"][1], got["eng"][1],
                                            return_indices=True)
            la = clusters(got["akk"][0][ia], args.seed)
            le = clusters(got["eng"][0][ie], args.seed)
            null = np.nan
            if cond == "mc32":
                null = float(np.nanmean([ari(la, rng.permutation(le))
                                         for _ in range(10)]))
            out[cond] = (ari(la, le), null)
        lines.append(f"| `{model.split('/')[-1]}` | {L} | {out['orig'][0]:.3f} | "
                     f"{out['mc32'][0]:.3f} | {out['mc32'][1]:+.3f} |")
        print(lines[-1], flush=True)

    # ---- V3: categories vs order -------------------------------------------
    lines += ["", "## V3 — time as categories vs time as order", "",
              "| model | layer | ARI(clusters, century) | centuries with order test "
              "| mean within-century rho |", "|---|---|---|---|---|"]
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    for _, row in peaks.iterrows():
        model, L = row.model, int(row.layer)
        vid, did = pick("akk", False)
        X, docs = grab(store, model, L, args.site, vid, did)
        if X is None:
            continue
        t = t_of.loc[docs].to_numpy()
        cent = (-(-t // 100)).astype(int)            # century number
        lab = clusters(X, args.seed)
        a_cat = ari(lab, cent)
        rhos = []
        for c in np.unique(cent):
            m = cent == c
            if m.sum() < 60 or len(np.unique(t[m])) < 5:
                continue
            oof_p, oof_t = [], []
            for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(X[m]):
                sc = StandardScaler().fit(X[m][tr])
                r = RidgeCV(alphas=np.logspace(-1, 4, 8)).fit(sc.transform(X[m][tr]), t[m][tr])
                oof_p.extend(r.predict(sc.transform(X[m][te]))); oof_t.extend(t[m][te])
            rhos.append(spearmanr(oof_p, oof_t).statistic)
        lines.append(f"| `{model.split('/')[-1]}` | {L} | {a_cat:.3f} | {len(rhos)} | "
                     f"{np.mean(rhos) if rhos else float('nan'):.3f} |")
        print(lines[-1], flush=True)

    # ---- V4: entity target sensitivity --------------------------------------
    lines += ["", "## V4 — name-probe target: first vs median vs last attested year", "",
              "Reign is a range; the probe target was the median attested year. "
              "Same activations, three target conventions.", "",
              "| arm | dataset | target=first | median | last |", "|---|---|---|---|---|"]
    WB = os.path.join("v_1", "src", "world_models")
    tgt = {k: (-corpus.groupby("ruler")["t"].agg(f)).to_dict()
           for k, f in (("first", "min"), ("median", "median"), ("last", "max"))}
    for arm in ("llama2_7b", "qwen3_8b", "thalesian_cunei400m"):
        for ds in ("assyrian_ruler_akk", "assyrian_ruler"):
            csv = os.path.join(WB, "data", "entity_datasets", f"{ds}.csv")
            adir = os.path.join(WB, "activations", arm, ds)
            files = sorted(glob.glob(os.path.join(adir, "ent_last.layer*.npz")),
                           key=lambda f: int(re.search(r"layer(\d+)", f).group(1)))
            if not (os.path.exists(csv) and files):
                continue
            df = pd.read_csv(csv)
            bare = (df.template == "bare").to_numpy()
            cells = []
            for key in ("first", "median", "last"):
                y_of = tgt[key]
                yb = df.name.map(y_of).to_numpy(float)
                best = -9
                for f in files[::max(1, len(files) // 8)]:
                    A = np.load(f)["acts"]
                    if len(A) != len(df):
                        break
                    Ab, ybb, ent = A[bare], yb[bare], df.entity_ix.to_numpy()[bare]
                    ok = np.isfinite(ybb)
                    Ab, ybb, ent = Ab[ok], ybb[ok], ent[ok]
                    rs = []
                    ents = np.unique(ent)
                    for d in range(60):
                        te_e = np.random.RandomState(d).choice(
                            ents, max(1, len(ents) // 5), replace=False)
                        m_te = np.isin(ent, te_e)
                        if m_te.sum() < 3 or (~m_te).sum() < 8:
                            continue
                        sc = StandardScaler().fit(Ab[~m_te])
                        r = RidgeCV(alphas=np.logspace(0, 4, 6)).fit(
                            sc.transform(Ab[~m_te]), ybb[~m_te])
                        p = r.predict(sc.transform(Ab[m_te]))
                        if len(np.unique(ybb[m_te])) > 1:
                            rs.append(spearmanr(p, ybb[m_te]).statistic)
                    if rs and np.nanmean(rs) > best:
                        best = float(np.nanmean(rs))
                cells.append(best)
            lines.append(f"| {arm} | {ds} | " +
                         " | ".join(f"{c:.3f}" for c in cells) + " |")
            print(lines[-1], flush=True)

    open(os.path.join(args.atlas_dir, "CONTROLS2.md"), "w").write("\n".join(lines) + "\n")
    print("wrote CONTROLS2.md")


if __name__ == "__main__":
    main()
