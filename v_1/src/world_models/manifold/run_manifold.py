"""Manifold analysis over the stored activations — 3 phases, one method at a time.

Phase 1  embed   : PCA(8) + UMAP(3) at the best layer -> coordinates saved (small).
Phase 2  isometry: Modell et al. — cosine vs squared feature distance (Chatterjee xi)
                   and graph-geodesic vs feature distance (Pearson rho), for several
                   feature metrics; plus the rho-vs-LAYER curve they never plot.
Phase 3  reduce  : Engels et al. — eps-mixture + separability indices per consecutive PC
                   pair, radius stats, and the centered singular-value spectrum.

Surfaces (see EXPERIMENT_MAP_MATRIX.md):
  fragment level  akk_maximal | eng_tier0  x  year | geo  x  last | mean      (cells B, C)
  entity level    the paper's six English datasets x last | mean             (cell A)

Coordinates + stats are written as small JSON/NPZ so every FIGURE can be regenerated
offline; only this step needs the (cluster-local, gitignored) activations.

    python run_manifold.py --method qwen3_8b --surface akk
    python run_manifold.py --method qwen3_8b --surface eng
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _WM)
sys.path.insert(0, os.path.join(_WM, "akkadian"))
import manifold_lib as ML                                  # noqa: E402
import reducibility as RD                                  # noqa: E402
from wm_lib import probing, entity_data                    # noqa: E402

AKK_ACTS = os.path.join(_WM, "akkadian", "activations")
ENG_ACTS = os.path.join(_WM, "activations")
AKK_LP = os.path.join(_WM, "akkadian", "results", "layers_pls")
ENG_PLS = os.path.join(_WM, "results", "eng_pls")
OUT = os.path.join(_HERE, "results")
KNN_K = 10
PCA_K = 8
MAX_N = 4000            # cap for the O(n^2) distance work


# columns used as the six colourings of the embedding figures (Fig 7.1 style)
FACETS = {
    "world_place":        ["latitude", "longitude", "country", "entity_subtype",
                           "page_views", "population"],
    "us_place":           ["latitude", "longitude", "state_name", "entity_subtype",
                           "page_views", "population"],
    "nyc_place":          ["latitude", "longitude", "borough_name", "facility_t_name"],
    "historical_figure":  ["death_year", "death_century", "country", "occupation",
                           "gender", "age"],
    "art":                ["release_date", "decade", "entity_type", "creator",
                           "page_views", "length"],
    "headline":           ["pub_date", "year", "section", "news_desk", "word_count"],
    "assyrian_ruler":     ["death_year", "name", "template", "n_texts"],
    "mesopotamian_place": ["longitude", "latitude", "region", "name", "template",
                           "n_texts"],
}


def _facet_frame(df, et):
    import pandas as pd
    cols = [c for c in FACETS.get(et, []) if c in df.columns]
    return df[cols].copy() if cols else None


def _layer_files(act_dir, site):
    return {int(re.search(r"layer(\d+)\.npz$", p).group(1)): p
            for p in glob.glob(os.path.join(act_dir, f"{site}.layer*.npz"))}


def _pca(X, k):
    from sklearn.decomposition import PCA
    k = min(k, min(X.shape) - 1)
    p = PCA(n_components=k, random_state=42)
    return p.fit_transform(X), p.explained_variance_ratio_.tolist()


def _umap(X, k=3):
    try:
        import umap
        return umap.UMAP(n_components=k, random_state=42).fit_transform(X).tolist()
    except Exception as e:                                    # noqa: BLE001
        print(f"    [umap skipped] {type(e).__name__}: {e}", flush=True)
        return None


def _pls2(X, y, k=2):
    """Supervised 2-D PLS coordinates — the projection used for the six-panel
    embedding figures (separation is partly baked in; read against the random arms)."""
    from sklearn.cross_decomposition import PLSRegression
    Y = y.reshape(-1, 1) if y.ndim == 1 else y
    try:
        return PLSRegression(n_components=k).fit_transform(X, Y)[0]
    except Exception as e:                                    # noqa: BLE001
        print(f"    [pls skipped] {type(e).__name__}: {e}", flush=True)
        return None


def analyse(tag, X_by_layer, best_layer, y, target, meta, args, facets=None):
    """Run the 3 phases for one surface and dump one JSON."""
    if best_layer not in X_by_layer:
        best_layer = sorted(X_by_layer)[len(X_by_layer) // 2]
    is_geo = (target == "geo")

    # subsample for the O(n^2) parts
    n0 = len(y)
    idx = np.arange(n0)
    if n0 > MAX_N:
        idx = np.random.RandomState(42).choice(n0, MAX_N, replace=False)
        idx.sort()

    Xb = X_by_layer[best_layer][idx]
    yb = y[idx]
    Xn, keep = ML.prep(Xb, r=None)
    yb = yb[keep]

    # ---------------- phase 1: embeddings ----------------
    P, evr = _pca(Xn, PCA_K)
    U = _umap(Xn, 3) if args.umap else None
    W = _pls2(Xn, yb, 2)                     # supervised 2-D, for the 6-panel figures

    # ---------------- phase 2: isometry ------------------
    if is_geo:
        metrics = {"haversine": ML.haversine_metric(yb)}
    else:
        metrics = {k: ML.year_metric(yb, kind=k) for k in ("abs", "log", "sqrt")}
    iso = {}
    for name, D in metrics.items():
        try:
            iso[name] = ML.isometry_stats(Xn, D, k=args.knn)
        except Exception as e:                                # noqa: BLE001
            iso[name] = {"error": f"{type(e).__name__}: {e}"[:120]}

    # rho vs layer (the extension they do not plot)
    Dbest = metrics["haversine" if is_geo else "abs"]
    per_layer = []
    for li in sorted(X_by_layer):
        try:
            Xl, kp = ML.prep(X_by_layer[li][idx], r=None)
            Dl = Dbest[np.ix_(kp, kp)]
            s = ML.isometry_stats(Xl, Dl, k=args.knn)
            per_layer.append({"layer": int(li), **{k: s[k] for k in ("xi_cos", "rho_geodesic")}})
        except Exception:                                     # noqa: BLE001
            per_layer.append({"layer": int(li), "xi_cos": float("nan"),
                              "rho_geodesic": float("nan")})

    # ---------------- phase 3: reducibility --------------
    red = RD.pair_indices(P, eps=args.eps, radius=args.radius)
    s_raw, s_cen = RD.centered_spectrum(Xn)

    out = {
        "tag": tag, "target": target, "best_layer": int(best_layer),
        "n": int(len(yb)), "n_total": int(n0), "knn_k": args.knn, **meta,
        "phase1": {"pca_explained_variance_ratio": evr},
        "phase2": {"isometry": iso, "per_layer": per_layer},
        "phase3": {"pair_indices": red, "svals_raw": s_raw, "svals_centered": s_cen},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(OUT, f"{tag}.coords.npz"),
                        pca=P.astype(np.float32),
                        umap=(np.array(U, dtype=np.float32) if U is not None
                              else np.zeros((0, 3), dtype=np.float32)),
                        pls=(np.asarray(W, dtype=np.float32) if W is not None
                             else np.zeros((0, 2), dtype=np.float32)),
                        y=yb.astype(np.float32))
    if facets is not None:
        f2 = facets.iloc[idx][keep] if hasattr(facets, "iloc") else None
        if f2 is not None:
            f2.to_csv(os.path.join(OUT, f"{tag}.facets.csv.gz"), index=False)
    b = iso.get("haversine" if is_geo else "abs", {})
    print(f"  [{tag}] L{best_layer} n={len(yb)} "
          f"xi={b.get('xi_cos', float('nan')):.3f} rho={b.get('rho_geodesic', float('nan')):.3f}",
          flush=True)
    return out


def run_akk(args):
    import akk_data as A
    df = A.load_fragments()
    # 6 rulers carry corrupt year values (7-10, ordinal not BCE) - drop them
    good_year = df.year.values >= 100
    for variant in ("akk_maximal", "eng_tier0"):
        act_dir = os.path.join(AKK_ACTS, args.method, variant)
        if not os.path.isdir(act_dir):
            continue
        for target in ("year", "geo"):
            tgt, valid = A.target_values(df, target)
            sel = valid & (good_year if target == "year" else np.ones(len(df), bool))
            for site in ("last", "mean"):
                files = _layer_files(act_dir, site)
                if not files:
                    continue
                bl = None
                bp = os.path.join(AKK_LP, args.method, f"{variant}.{target}.{site}.json")
                if os.path.exists(bp):
                    bl = json.load(open(bp)).get("best_layer")
                Xs = {}
                for li, p in files.items():
                    X = np.load(p)["acts"][sel].astype(np.float32)
                    X, bad = probing.sanitize(X)
                    if bad <= 0.01:
                        Xs[li] = X
                if not Xs:
                    continue
                analyse(f"akk__{args.method}__{variant}__{target}__{site}",
                        Xs, bl, tgt[sel], target,
                        {"method": args.method, "variant": variant, "site": site,
                         "level": "fragment"}, args)


def run_eng(args):
    for et in entity_data.ENTITY_TYPES:
        act_dir = os.path.join(ENG_ACTS, args.method, et)
        if not os.path.isdir(act_dir):
            continue
        df = entity_data.load_entity_df(et)
        tgt, valid = entity_data.target_values(et, df)
        _, is_place = entity_data.FEATURES[et]
        meta = json.load(open(os.path.join(act_dir, "metadata.json")))
        n = meta["n_rows"]
        tgt, valid = tgt[:n], valid[:n]
        for site in ("last", "mean"):
            files = _layer_files(act_dir, site)
            if not files:
                continue
            bl = None
            bp = os.path.join(ENG_PLS, args.method, f"{et}.{site}.json")
            if os.path.exists(bp):
                bl = json.load(open(bp)).get("best_layer")
            Xs = {}
            for li, p in files.items():
                X = np.load(p)["acts"][:n][valid].astype(np.float32)
                X, bad = probing.sanitize(X)
                if bad <= 0.01:
                    Xs[li] = X
            if not Xs:
                continue
            analyse(f"eng__{args.method}__{et}__{site}", Xs, bl, tgt[valid],
                    "geo" if is_place else "year",
                    {"method": args.method, "entity_type": et, "site": site,
                     "level": "entity"}, args,
                    facets=_facet_frame(df.iloc[:n][valid].reset_index(drop=True), et))



def run_ent(args):
    """Cell B at ENTITY level: Assyrian rulers / Mesopotamian places written in English,
    in the paper's short-entity format plus carrier sentences. Four pooling sites, of
    which `ent_last` is the paper-faithful 'last token of the entity name'."""
    import pandas as pd
    sys.path.insert(0, os.path.join(_WM, "akkadian"))
    import probe_entity as PE
    for et, (feat, is_place) in PE.ENTITY_TYPES.items():
        act_dir = os.path.join(PE.ACTS_DIR, args.method, et)
        if not os.path.isdir(act_dir):
            continue
        df = PE.load_df(et)
        y, _ = PE.targets(et, df)
        for site in PE.SITES:
            files = _layer_files(act_dir, site)
            if not files:
                continue
            bl = None
            bp = os.path.join(PE.RESULTS_DIR, "probes_entity", args.method,
                              f"{et}.{site}.json")
            if os.path.exists(bp):
                bl = json.load(open(bp)).get("best_layer")
            Xs = {}
            for li, p in files.items():
                X = np.load(p)["acts"].astype(np.float32)
                X, bad = probing.sanitize(X)
                if bad <= 0.01:
                    Xs[li] = X
            if not Xs:
                continue
            analyse(f"ent__{args.method}__{et}__{site}", Xs, bl, y,
                    "geo" if is_place else "year",
                    {"method": args.method, "entity_type": et, "site": site,
                     "level": "entity_cellB"}, args, facets=_facet_frame(df, et))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--surface", default="akk", choices=["akk", "eng", "ent", "all"])
    ap.add_argument("--knn", type=int, default=KNN_K)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--umap", action="store_true", default=True)
    ap.add_argument("--no-umap", dest="umap", action="store_false")
    args = ap.parse_args()
    if args.surface in ("akk", "all"):
        run_akk(args)
    if args.surface in ("eng", "all"):
        run_eng(args)
    if args.surface in ("ent", "all"):
        run_ent(args)


if __name__ == "__main__":
    main()
