"""E6 — cluster structure of the embeddings: does unsupervised clustering find
our metadata, and is k=8 (the ruler count) ever the optimal k?

Per model x cleaning {maximal, engtier0} at the model's best year layer
(l2-normalized mean acts; method 'tfidf' builds char_wb(2,5)->SVD-512 features
from the texts instead):

FULL-CORPUS pass
  * KMeans k in 2..15 (n_init=10, seed 42): silhouette(k), Davies-Bouldin(k),
    Calinski-Harabasz(k), inertia(k); best k by silhouette.
  * At k=8: ARI + AMI of the cluster labels vs each metadata partition —
    ruler8 (canonical 8 only), period, sub_genre (top-8), provenance (top-8),
    year quartiles.
  * Label silhouettes: silhouette of the METADATA labelings themselves in the
    embedding space (how compact is each partition, no clustering involved).

MC pass (the deck's 200 balanced draws, 8 rulers x 21)
  * per draw: KMeans k=8 -> ARI vs ruler, silhouette@8; best k by silhouette
    over 2..12. Report mean +- std and the distribution of chosen k
    ("is 8 optimal?" under the balanced protocol).

Output: results/e6_clusters__<method>.json (committed).
Usage:  python run_cluster_metrics.py --method qwen3_32b
        python run_cluster_metrics.py --method tfidf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
sys.path.insert(0, str(_THIS.parents[1] / "eda"))
from geo_loader import find_acts_dir, load_layer, available_layers          # noqa: E402
from dump_stress_coords import best_layer_maximal, best_layer_translation  # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"
CLEANINGS = ["maximal", "engtier0"]
K_FULL = list(range(2, 16))
K_MC = list(range(2, 13))


def tfidf_features(df, cleaning):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    if cleaning == "maximal":
        texts = df["text_maximal"].fillna("").astype(str).tolist()
    else:
        tr = pd.read_parquet(TRANSLATIONS).set_index("fragment_id")["eng_tier0"]
        texts = tr.reindex(df["fragment_id"].astype(str)).fillna("").astype(str).tolist()
    X = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit_transform(texts))
    return TruncatedSVD(512, random_state=0).fit_transform(X).astype(np.float32)


def top_or_other(series, topn=8):
    top = series.value_counts().head(topn).index
    return np.where(series.isin(top), series.astype(str), "OTHER")


def run(args):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    from sklearn.preprocessing import normalize

    df = pd.read_parquet(CORPUS)
    fids = json.load(open(BAL / "corpus_fragment_order.json"))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    year = df["year"].to_numpy(dtype=float)
    ruler = df["ruler"].astype(str).to_numpy()
    canon8 = df["ruler"].value_counts().head(8).index.tolist()
    meta = {
        "ruler8": np.where(np.isin(ruler, canon8), ruler, "OTHER"),
        "period": top_or_other(df["period"], 4),
        "sub_genre": top_or_other(df["sub_genre"], 8),
        "provenance": top_or_other(df["provenance"], 8),
    }
    dm = np.load(BAL / "draws_matrix.npy")

    out = {"method": args.method, "protocol": "e6_cluster_metrics",
           "canon8": canon8, "cleanings": {}}
    fp = _THIS.parent / "results" / f"e6_clusters__{args.method}.json"
    if fp.exists():
        out["cleanings"] = json.loads(fp.read_text()).get("cleanings", {})

    for cl in CLEANINGS:
        if cl in out["cleanings"] and not out["cleanings"][cl].get("missing"):
            print(f"[{args.method} x {cl}] already done, skip"); continue
        t0 = time.time()
        if args.method == "tfidf":
            X = tfidf_features(df, cl); bl = -1
        else:
            d = find_acts_dir(args.method, cl, "mean")
            if d is None:
                print(f"[{args.method} x {cl}] acts missing")
                out["cleanings"][cl] = {"missing": True}; continue
            bl = (best_layer_maximal(args.method) if cl == "maximal"
                  else best_layer_translation(args.method, cl))
            if bl not in available_layers(d):
                bl = 0
            X = np.nan_to_num(load_layer(d, bl).astype(np.float64))
        Xn = normalize(X)

        # ---- full-corpus pass ----
        kcurve = {}
        labels_at = {}
        for k in K_FULL:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xn)
            labels_at[k] = km.labels_
            kcurve[str(k)] = {
                "silhouette": float(silhouette_score(Xn, km.labels_)),
                "davies_bouldin": float(davies_bouldin_score(Xn, km.labels_)),
                "calinski_harabasz": float(calinski_harabasz_score(Xn, km.labels_)),
                "inertia": float(km.inertia_)}
        best_k = max(K_FULL, key=lambda k: kcurve[str(k)]["silhouette"])
        lab8 = labels_at[8]
        align8 = {}
        for name, lab in meta.items():
            m = lab != "OTHER" if name == "ruler8" else np.ones(len(lab), bool)
            align8[name] = {"ari": float(adjusted_rand_score(lab[m], lab8[m])),
                            "ami": float(adjusted_mutual_info_score(lab[m], lab8[m]))}
        okY = np.isfinite(year)
        yq = np.digitize(year[okY], np.nanpercentile(year[okY], [25, 50, 75]))
        align8["year_quartile"] = {
            "ari": float(adjusted_rand_score(yq, lab8[okY])),
            "ami": float(adjusted_mutual_info_score(yq, lab8[okY]))}
        label_sil = {}
        for name, lab in meta.items():
            m = lab != "OTHER"
            if len(set(lab[m])) > 1 and m.sum() > 50:
                label_sil[name] = float(silhouette_score(Xn[m], lab[m]))

        # ---- MC 200-draw pass ----
        ari_r, sil8, bestks = [], [], []
        for di in range(dm.shape[0]):
            rows = np.where(dm[di])[0]
            Z = Xn[rows]; r = ruler[rows]
            km8 = KMeans(n_clusters=8, n_init=5, random_state=42).fit(Z)
            ari_r.append(adjusted_rand_score(r, km8.labels_))
            sil8.append(silhouette_score(Z, km8.labels_))
            sc = {k: silhouette_score(Z, KMeans(n_clusters=k, n_init=5,
                                                random_state=42).fit_predict(Z))
                  for k in K_MC}
            bestks.append(max(sc, key=sc.get))
        out["cleanings"][cl] = {
            "best_layer": int(bl), "k_curve": kcurve,
            "best_k_full_silhouette": int(best_k),
            "kmeans8_vs_metadata": align8, "label_silhouettes": label_sil,
            "mc": {"ari_ruler_mean": float(np.mean(ari_r)),
                   "ari_ruler_std": float(np.std(ari_r)),
                   "sil_k8_mean": float(np.mean(sil8)),
                   "best_k_counts": dict(Counter(bestks)),
                   "n_draws": int(dm.shape[0])}}
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[{args.method} x {cl}] L{bl} best_k={best_k} "
              f"ARI(ruler8,k8)={align8['ruler8']['ari']:.3f} "
              f"MC ARI={np.mean(ari_r):.3f}+-{np.std(ari_r):.3f} "
              f"MC best-k mode={Counter(bestks).most_common(1)[0]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    run(p.parse_args())
