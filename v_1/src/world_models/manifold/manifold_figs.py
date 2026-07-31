"""Faithful Modell et al. (arXiv 2505.18235) manifold figures on FULL activations.

`run_manifold.py` saved summary statistics and PCA-8 coordinates; that is enough for the
numbers but NOT for the paper's figures, which are computed on the representations
themselves. This script runs their pipeline end-to-end on the full hidden states and
renders the three figures they publish, plus saves the underlying arrays so the styling
can be reworked offline without touching the cluster.

Their pipeline, reproduced exactly:
  1. drop rows with norm <= 1e-3
  2. L2 row-normalise
  3. (high-d only) project onto the top-r UNCENTERED singular directions via svds,
     keep u*s, re-normalise rows            [their cells 32 / 40]
  4. kNN graph, Euclidean edge weights, symmetrised; clamp weights < 1e-3
  5. Dijkstra -> the manifold metric; restrict to the largest connected component
  6. cosine similarity on the same vectors
  7. two diagnostics:
       xi  = Chatterjee(feature distance, cosine similarity)     [plotted vs SQUARED d]
       rho = Pearson(feature distance, graph-geodesic distance)
  8. PCA(5) for the arc pictures - the structure often lives in PC3-PC4, not PC1-PC2

Feature metrics: |dyear| plus their log-recency reparameterisation; haversine for geo.

    python manifold_figs.py --method llama2_70b --surface ent
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _WM)
sys.path.insert(0, os.path.join(_WM, "akkadian"))
import manifold_lib as ML                       # noqa: E402
from wm_lib import probing, entity_data         # noqa: E402

OUT = os.path.join(_HERE, "figs")
MAXN = 2500          # pair matrices are O(n^2); their year set is a similar size


def _layers(act_dir, site):
    return {int(re.search(r"layer(\d+)\.npz$", p).group(1)): p
            for p in glob.glob(os.path.join(act_dir, f"{site}.layer*.npz"))}


def _load(path, rows=None):
    X = np.load(path)["acts"]
    if rows is not None:
        X = X[rows]
    X, _ = probing.sanitize(X.astype(np.float32))
    return X


# ------------------------------------------------------------------ the three figures
def figures(tag, X, y, is_geo, args):
    os.makedirs(OUT, exist_ok=True)
    if len(X) > args.max_n:
        s = np.random.RandomState(0).choice(len(X), args.max_n, replace=False)
        s.sort(); X, y = X[s], y[s]

    Xn, keep = ML.prep(X, r=args.rank)          # steps 1-3, on the FULL vectors
    y = y[keep]
    Dm, kp, ncomp = ML.manifold_distance(Xn, args.knn)      # steps 4-5
    cos = ML.cosine_similarity_matrix(Xn)                    # step 6

    mets = ({"haversine": ML.haversine_metric(y)} if is_geo
            else {k: ML.year_metric(y, kind=k) for k in ("abs", "log")})

    # ---- fig 1: the arc, PCA pairs coloured by the target
    from sklearn.decomposition import PCA
    P = PCA(n_components=min(6, min(Xn.shape) - 1), random_state=42).fit_transform(Xn)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4.3))
    c = y[:, 0] if (is_geo and y.ndim > 1) else y
    for ax, (a, b) in zip(axes, pairs):
        if max(a, b) >= P.shape[1]:
            ax.axis("off"); continue
        sc = ax.scatter(P[:, a], P[:, b], c=c, s=7, cmap="viridis", alpha=.85, linewidths=0)
        ax.set_xlabel(f"PCA axis {a+1}"); ax.set_ylabel(f"PCA axis {b+1}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=axes[-1], fraction=.046)
    fig.suptitle(f"{tag} — representation coloured by target (full activations, "
                 f"rank={args.rank}, n={len(Xn)})", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}__arc.png", dpi=130); plt.close(fig)

    # ---- figs 2-3: the two isometry diagnostics, one column per feature metric
    stats = {}
    fig, axes = plt.subplots(2, len(mets), figsize=(5.6 * len(mets), 9), squeeze=False)
    for ci, (name, D) in enumerate(mets.items()):
        iu = np.triu_indices(len(Xn), 1)
        cy = np.repeat(c, len(c)).reshape(len(c), len(c))[iu]
        xi = ML.chatterjee_corr(D[iu], cos[iu])
        ax = axes[0][ci]
        ax.scatter(D[iu] ** 2, cos[iu], c=cy, s=.4, alpha=.10, cmap="viridis", linewidths=0)
        ax.set_xlabel(f"Squared {name} distance"); ax.set_ylabel("Cosine similarity")
        ax.text(.97, .06, f"$\\xi = {xi:.3f}$", transform=ax.transAxes, ha="right",
                fontsize=12, bbox=dict(fc="w", ec="k"))
        Dk = D[np.ix_(kp, kp)]; iu2 = np.triu_indices(int(kp.sum()), 1)
        ck = c[kp]; cy2 = np.repeat(ck, len(ck)).reshape(len(ck), len(ck))[iu2]
        fin = np.isfinite(Dm[iu2])
        rho = ML.pearson(Dk[iu2][fin], Dm[iu2][fin])
        ax = axes[1][ci]
        ax.scatter(Dk[iu2][fin], Dm[iu2][fin], c=cy2[fin], s=.4, alpha=.10,
                   cmap="viridis", linewidths=0)
        ax.set_xlabel(f"{name} distance"); ax.set_ylabel("Manifold (graph-geodesic) distance")
        ax.text(.97, .06, f"$\\rho = {rho:.3f}$", transform=ax.transAxes, ha="right",
                fontsize=12, bbox=dict(fc="w", ec="k"))
        stats[name] = {"xi_cos": xi, "rho_geodesic": rho}
    fig.suptitle(f"{tag} — Modell isometry diagnostics (k={args.knn}, "
                 f"{ncomp} component(s), n={int(kp.sum())})", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}__isometry.png", dpi=130); plt.close(fig)

    np.savez_compressed(f"{OUT}/{tag}__arrays.npz", pca=P.astype(np.float32),
                        y=np.asarray(y, dtype=np.float32))
    with open(f"{OUT}/{tag}__stats.json", "w") as f:
        json.dump({"tag": tag, "n": int(len(Xn)), "rank": args.rank, "knn": args.knn,
                   "n_components": int(ncomp), "stats": stats}, f, indent=2)
    print(f"  [{tag}] " + " ".join(f"{k}: xi={v['xi_cos']:+.3f} rho={v['rho_geodesic']:+.3f}"
                                   for k, v in stats.items()), flush=True)


# ------------------------------------------------------------------------- surfaces
def run_ent(args):
    import probe_entity as PE
    for et, (_, is_place) in PE.ENTITY_TYPES.items():
        d = os.path.join(PE.ACTS_DIR, args.method, et)
        if not os.path.isdir(d):
            continue
        df = PE.load_df(et); y, _ = PE.targets(et, df)
        bare = (df["template"].values == "bare")
        for site in ("ent_last", "ent_mean"):
            fs = _layers(d, site)
            if not fs:
                continue
            bp = os.path.join(PE.RESULTS_DIR, "probes_entity", args.method, f"{et}.{site}.json")
            bl = json.load(open(bp)).get("best_layer") if os.path.exists(bp) else None
            if bl not in fs:
                bl = sorted(fs)[len(fs) // 2]
            X = _load(fs[bl])
            figures(f"ent__{args.method}__{et}__{site}__bare", X[bare], y[bare], is_place, args)


def run_eng(args):
    for et in entity_data.ENTITY_TYPES:
        d = os.path.join(_WM, "activations", args.method, et)
        if not os.path.isdir(d):
            continue
        df = entity_data.load_entity_df(et)
        tgt, valid = entity_data.target_values(et, df)
        _, is_place = entity_data.FEATURES[et]
        n = json.load(open(os.path.join(d, "metadata.json")))["n_rows"]
        tgt, valid = tgt[:n], valid[:n]
        for site in ("last", "mean"):
            fs = _layers(d, site)
            if not fs:
                continue
            bp = os.path.join(_WM, "results", "eng_pls", args.method, f"{et}.{site}.json")
            bl = json.load(open(bp)).get("best_layer") if os.path.exists(bp) else None
            if bl not in fs:
                bl = sorted(fs)[len(fs) // 2]
            X = _load(fs[bl])[:n][valid]
            figures(f"eng__{args.method}__{et}__{site}", X, tgt[valid], is_place, args)


def run_akk(args):
    import akk_data as A
    df = A.load_fragments()
    good = df.year.values >= 100
    for variant in ("akk_maximal", "eng_tier0"):
        d = os.path.join(_WM, "akkadian", "activations", args.method, variant)
        if not os.path.isdir(d):
            continue
        for target in ("year", "geo"):
            tgt, valid = A.target_values(df, target)
            sel = valid & (good if target == "year" else np.ones(len(df), bool))
            for site in ("last", "mean"):
                fs = _layers(d, site)
                if not fs:
                    continue
                bp = os.path.join(_WM, "akkadian", "results", "layers_pls", args.method,
                                  f"{variant}.{target}.{site}.json")
                bl = json.load(open(bp)).get("best_layer") if os.path.exists(bp) else None
                if bl not in fs:
                    bl = sorted(fs)[len(fs) // 2]
                X = _load(fs[bl], rows=sel)
                figures(f"akk__{args.method}__{variant}__{target}__{site}",
                        X, tgt[sel], target == "geo", args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--surface", default="all", choices=["ent", "eng", "akk", "all"])
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--rank", type=int, default=4,
                    help="uncentered SVD rank before the graph (their denoising step)")
    ap.add_argument("--max-n", type=int, default=MAXN)
    args = ap.parse_args()
    for s, fn in (("ent", run_ent), ("eng", run_eng), ("akk", run_akk)):
        if args.surface in (s, "all"):
            try:
                fn(args)
            except Exception as e:                              # noqa: BLE001
                print(f"[error] {s}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
