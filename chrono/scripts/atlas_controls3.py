"""Controls wave 3 (advisor follow-ups, 2026-10-06).

V1b MASK-ONLY. Wave 2's [mask_ruler + crop32] control removed names AND
    length together, so the collapse of the LLMs' alignment could not be
    attributed. This isolates the name: rerun the language alignment on
    views with royal names masked but the FULL text kept. If alignment
    survives mask-only but dies under crop, length/full-content carried it;
    if it dies already here, the names carried it.
V3b ORDER-TEST DIAGNOSTICS. The within-century rho (.62-.70) is only
    meaningful if each tested century holds many distinct years and no
    single year dominates. Report, per tested century: n docs, n distinct
    years, the majority-year share, and the rho again.

Everything runs from stored artifacts; CPU only.
Appends to chrono/reports/tier0/atlas/CONTROLS3.md.
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
    lines = ["# Controls wave 3 — mask-only alignment, order-test diagnostics", ""]

    store = EmbStore(os.path.join(args.art, "emb_store"))
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    views = pd.read_parquet(os.path.join(args.art, "views.parquet"))
    aug = views["augs"].fillna("").astype(str)
    t_of = corpus.set_index("doc_id")["t"].astype(float)

    # what masked-but-uncropped conditions exist at all?
    inv = views.assign(a=aug).groupby(["a", "lang"]).size()
    lines += ["Stored view conditions (augs x lang x count):", "```",
              inv.to_string(), "```", ""]
    print(inv.to_string(), flush=True)

    mask_only = aug.str.contains("mask_ruler") & ~aug.str.contains("crop")

    def pick(lang, cond):
        m = {"orig": aug == "", "mask": mask_only}[cond] & (views.lang == lang)
        sub = views[m].drop_duplicates("doc_id")
        return sub.view_id.to_numpy(), sub.doc_id.to_numpy()

    B = pd.read_csv(os.path.join(args.atlas_dir, "B_language_alignment.csv"))
    peaks = B.loc[B.groupby("model").lang_ari.idxmax(), ["model", "layer", "lang_ari"]]

    # ---- V1b: alignment with names masked, full length ---------------------
    lines += ["## V1b — language alignment on [mask_ruler], full text", "",
              "| model | layer | ARI orig | ARI mask-only | shuffle null |",
              "|---|---|---|---|---|"]
    if mask_only.sum() == 0:
        lines += ["", "No mask-only (uncropped) views are stored; condition "
                  "requires a new extraction."]
        print("[V1b] no mask-only views stored", flush=True)
    else:
        for _, row in peaks.iterrows():
            model, L = row.model, int(row.layer)
            out = {}
            for cond in ("orig", "mask"):
                got = {}
                for lang in ("akk", "eng"):
                    vid, did = pick(lang, cond)
                    if len(vid) == 0:
                        got = None; break
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
                if cond == "mask":
                    null = float(np.nanmean([ari(la, rng.permutation(le))
                                             for _ in range(10)]))
                out[cond] = (ari(la, le), null)
            lines.append(f"| `{model.split('/')[-1]}` | {L} | {out['orig'][0]:.3f} | "
                         f"{out['mask'][0]:.3f} | {out['mask'][1]:+.3f} |")
            print(lines[-1], flush=True)

    # ---- V3b: what is inside each tested century? ---------------------------
    lines += ["", "## V3b — order-test diagnostics per century", "",
              "Only centuries with >= 60 docs and >= 5 distinct years were "
              "tested in wave 2. What do they hold?", "",
              "| model | century (BC) | n docs | distinct years | majority-year "
              "share | rho |", "|---|---|---|---|---|---|"]
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    for _, row in peaks.iterrows():
        model, L = row.model, int(row.layer)
        sub = views[(aug == "") & (views.lang == "akk")].drop_duplicates("doc_id")
        X, docs = grab(store, model, L, args.site,
                       sub.view_id.to_numpy(), sub.doc_id.to_numpy())
        if X is None:
            continue
        t = t_of.loc[docs].to_numpy()
        cent = (-(-t // 100)).astype(int)
        for c in np.unique(cent):
            m = cent == c
            uniq, cnt = np.unique(t[m], return_counts=True)
            if m.sum() < 60 or len(uniq) < 5:
                continue
            oof_p, oof_t = [], []
            for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(X[m]):
                sc = StandardScaler().fit(X[m][tr])
                r = RidgeCV(alphas=np.logspace(-1, 4, 8)).fit(
                    sc.transform(X[m][tr]), t[m][tr])
                oof_p.extend(r.predict(sc.transform(X[m][te])))
                oof_t.extend(t[m][te])
            rho = spearmanr(oof_p, oof_t).statistic
            lines.append(f"| `{model.split('/')[-1]}` | {-c} | {int(m.sum())} | "
                         f"{len(uniq)} | {cnt.max() / m.sum():.0%} | {rho:.3f} |")
            print(lines[-1], flush=True)

    open(os.path.join(args.atlas_dir, "CONTROLS3.md"), "w").write("\n".join(lines) + "\n")
    print("wrote CONTROLS3.md")


if __name__ == "__main__":
    main()
