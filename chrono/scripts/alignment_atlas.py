"""P6 figures — the alignment atlas over the 1,193 dated inscriptions.

Modelled on the figures of arXiv:2610.01821 (NLMCD/CBA), with our questions:

Fig A  layer x layer manifold alignment, one model. Per layer: UMAP ->
       HDBSCAN clusters over the Akkadian clean-view embeddings; between
       layers: adjusted Rand on non-noise overlap. Their Fig. 3/4 analogue —
       where in depth does representational structure form and persist?
Fig B  language alignment per layer. The same documents carry an Akkadian
       and an English view; per layer we cluster each language separately
       and measure ARI across languages on the matched documents. High =
       the manifolds are about CONTENT (survives translation); low = about
       SCRIPT/SURFACE. Plus period-silhouette per language per layer.
Fig C  depth of the time signal. Ridge rho(t) per layer per language,
       pooled over ruler-grouped folds — "when is time born, per language".
       If entity layerwise results exist (WB / WB-akk), their best curves
       are overlaid so name-level and document-level sit on one figure.

Everything is driven by what the tier0 store actually holds (cunei400m all
layers; llama/qwen a 4-layer ladder): missing cells are skipped, not fatal.
Outputs: chrono/reports/tier0/atlas/*.png + ATLAS.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.models.store import EmbStore                        # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402


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


def pooled_rho(X, t, folds):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    pred, truth = [], []
    for f in folds:
        tr = [i for i in f["train_ix"]]; te = [i for i in f["test_ix"]]
        sc = StandardScaler().fit(X[tr])
        r = RidgeCV(alphas=np.logspace(-1, 4, 10)).fit(sc.transform(X[tr]), t[tr])
        s = r.predict(sc.transform(X[te]))
        pred.extend(s - s.mean()); truth.extend(t[te])
    return spearmanr(pred, truth).statistic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--atlas-model", default="Thalesian/cuneiformBase-400m",
                    help="the model for the layer x layer panel (needs many layers)")
    ap.add_argument("--site", default="mean")
    ap.add_argument("--out-dir", default="chrono/reports/tier0/atlas")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    store = EmbStore(os.path.join(args.art, "emb_store"))
    man = store.manifest()
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    views = pd.read_parquet(os.path.join(args.art, "views.parquet"))
    clean = views[views["augs"].fillna("") == ""].drop_duplicates(["doc_id", "lang"])
    with open(os.path.join(args.art, "splits", "gkf_ruler.json")) as f:
        gkf = json.load(f)
    t_of = corpus.set_index("doc_id")["t"].astype(float)
    # century labels for silhouettes (periods are too coarse within one corpus)
    cent = ((-t_of // 100).astype(int)).rename("century")

    def fetch(model, layer, lang):
        sub = clean[clean.lang == lang]
        ids = sub.view_id.tolist()
        if not np.all(store.has(model, int(layer), args.site, ids)):
            return None, None
        X = store.get(model, int(layer), args.site, ids).astype(np.float32)
        return X, sub.doc_id.to_numpy()

    lines = ["# Alignment atlas — dated inscriptions, tier0 store", ""]
    layers_of = {m: sorted(g.layer.unique())
                 for m, g in man[man.site == args.site].groupby("model")}

    # ---- Fig A: layer x layer manifold alignment -------------------------
    am = args.atlas_model
    labs, Ls = {}, layers_of.get(am, [])
    for L in Ls:
        X, docs = fetch(am, L, "akk")
        if X is not None:
            labs[L] = clusters(X, args.seed)
    Ls = sorted(labs)
    if len(Ls) >= 3:
        M = np.full((len(Ls), len(Ls)), np.nan)
        for i, a in enumerate(Ls):
            for j, b in enumerate(Ls):
                M[i, j] = 1.0 if i == j else ari(labs[a], labs[b])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, origin="lower", cmap="magma", vmin=0)
        ax.set_xticks(range(len(Ls)), Ls); ax.set_yticks(range(len(Ls)), Ls)
        ax.set_xlabel("layer"); ax.set_ylabel("layer")
        ax.set_title(f"Manifold alignment (ARI) — {am.split('/')[-1]}, akk")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "A_layer_alignment.png"), dpi=150)
        plt.close(fig)
        lines += [f"**Fig A** `A_layer_alignment.png` — {am}, {len(Ls)} layers, "
                  "UMAP+HDBSCAN per layer, ARI between layers.", ""]

    # ---- Fig B: language alignment + century silhouette per layer --------
    from sklearn.metrics import silhouette_score
    rowsB = []
    for model, Ls_m in layers_of.items():
        for L in Ls_m:
            Xa, da = fetch(model, L, "akk")
            Xe, de = fetch(model, L, "eng")
            if Xa is None or Xe is None:
                continue
            common_docs, ia, ie = np.intersect1d(da, de, return_indices=True)
            la = clusters(Xa[ia], args.seed); le = clusters(Xe[ie], args.seed)
            y = cent.loc[common_docs].to_numpy()
            rowsB.append(dict(model=model, layer=int(L), lang_ari=ari(la, le),
                              sil_akk=silhouette_score(Xa[ia], y),
                              sil_eng=silhouette_score(Xe[ie], y)))
    B = pd.DataFrame(rowsB)
    if len(B):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for model, g in B.groupby("model"):
            g = g.sort_values("layer")
            axes[0].plot(g.layer, g.lang_ari, marker="o", label=model.split("/")[-1])
            axes[1].plot(g.layer, g.sil_akk, marker="o", label=f"{model.split('/')[-1]} akk")
            axes[1].plot(g.layer, g.sil_eng, marker="s", ls="--", label=f"{model.split('/')[-1]} eng")
        axes[0].set_title("akk↔eng manifold alignment (ARI)"); axes[0].set_xlabel("layer")
        axes[1].set_title("century silhouette"); axes[1].set_xlabel("layer")
        for ax in axes: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "B_language_alignment.png"), dpi=150)
        plt.close(fig)
        B.to_csv(os.path.join(args.out_dir, "B_language_alignment.csv"), index=False)
        lines += ["**Fig B** `B_language_alignment.png` — high ARI = manifolds encode "
                  "content (survive translation); low = script/surface. Right panel: "
                  "century separation per language per layer.", ""]

    # ---- Fig C: depth of the time signal ---------------------------------
    folds = []
    for f in gkf["folds"]:
        folds.append({"train_ix": None, "test_ix": None, "train": f["train"], "test": f["test"]})
    rowsC = []
    for model, Ls_m in layers_of.items():
        for L in Ls_m:
            for lang in ("akk", "eng"):
                X, docs = fetch(model, L, lang)
                if X is None:
                    continue
                pos = {d: i for i, d in enumerate(docs)}
                fl = [{"train_ix": [pos[d] for d in f["train"] if d in pos],
                       "test_ix": [pos[d] for d in f["test"] if d in pos]} for f in gkf["folds"]]
                t = t_of.loc[docs].to_numpy()
                rowsC.append(dict(model=model, layer=int(L), lang=lang,
                                  rho=pooled_rho(X, t, fl)))
    C = pd.DataFrame(rowsC)
    if len(C):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (model, lang), g in C.groupby(["model", "lang"]):
            g = g.sort_values("layer")
            ax.plot(g.layer, g.rho, marker="o" if lang == "akk" else "s",
                    ls="-" if lang == "akk" else "--",
                    label=f"{model.split('/')[-1]} {lang}")
        # overlay entity depth curves if the WB layerwise summaries exist
        ent = os.path.join("v_1", "src", "world_models", "akkadian", "results",
                           "summary_entity_layerwise.csv")
        if os.path.exists(ent):
            E = pd.read_csv(ent)
            for ds in [d for d in E.get("dataset", pd.Series()).unique()
                       if isinstance(d, str) and d.startswith("assyrian_ruler")]:
                g = E[(E.dataset == ds) & (E.get("site") == "ent_last")]
                col = next((c for c in ("mc_rho", "rho", "MC rho") if c in g.columns), None)
                arm = next((a for a in ("llama2_7b", "qwen3_8b") if (g.get("arm") == a).any()), None)
                if col and arm is not None:
                    gg = g[g.arm == arm].sort_values("layer")
                    ax.plot(gg.layer, gg[col], marker="^", ls=":",
                            label=f"ENTITY {ds.replace('assyrian_ruler', 'name')} {arm}")
        ax.set_xlabel("layer"); ax.set_ylabel("pooled Spearman rho (gkf)")
        ax.set_title("Where in depth does time live?")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "C_time_depth.png"), dpi=150)
        plt.close(fig)
        C.to_csv(os.path.join(args.out_dir, "C_time_depth.csv"), index=False)
        lines += ["**Fig C** `C_time_depth.png` — document-level rho per layer per "
                  "language; entity-level curves overlaid where available.", ""]

    open(os.path.join(args.out_dir, "ATLAS.md"), "w").write("\n".join(lines) + "\n")
    print(f"atlas written to {args.out_dir}")


if __name__ == "__main__":
    main()
