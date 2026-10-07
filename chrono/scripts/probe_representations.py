"""probe_representations.py — S1 of PLAN_SCALE_SSL: is an embedding space
meaningful for TIME without any dates?

For one embedding source (EmbStore model/layer/site over corpus_all ids
'ssl::<fragment_id>'), on every label set with enough support:

  probes      linear (logistic) and MLP balanced accuracy for PERIOD, GENRE,
              PROVENANCE and SOURCE, tablet-level 5-fold CV, classes >= min_n
              docs; SOURCE is the bias indicator (easy source + hard period =
              the model learned corpora, not time)
  within-src  the period probe repeated inside each source that has >= 2
              periods (period is correlated with source: OB ~ archibab,
              LB ~ lbl_letters)
  held-out    period probe trained on all sources but one, tested on it
  geometry    silhouette score of the label in the raw embedding space and in
              a 2-d UMAP, each against a 200x label-permutation null
  retrieval   k-NN period purity (k = 10)

Writes one markdown per (model, layer, site) under --out-dir and appends rows
to results.parquet (run_id 's1_probe::<model>::L<layer>::<site>').
"""
from __future__ import annotations
import argparse, json, os, re, sys, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402
from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, silhouette_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold             # noqa: E402
from sklearn.neighbors import NearestNeighbors                  # noqa: E402
from sklearn.decomposition import PCA                           # noqa: E402

warnings.filterwarnings("ignore")


def _safe(name: str) -> str:
    """Filename-safe model tag. The store's model names carry '::' and '/'
    (e.g. 'ssl::ssl_barlow_cunei400m-s0'), which the cluster filesystem
    rejects: every C10 probe computed its whole battery and then died with
    OSError on the output path (jobs 33691+)."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def cv_probe(X, y, kind, seed=0, folds=5):
    le = LabelEncoder().fit(y); yi = le.transform(y)
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yi):
        sc = StandardScaler().fit(X[tr])
        clf = (LogisticRegression(max_iter=3000, C=0.5) if kind == "linear"
               else MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=seed))
        clf.fit(sc.transform(X[tr]), yi[tr])
        accs.append(balanced_accuracy_score(yi[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(accs)), float(np.std(accs)), len(le.classes_)


def silhouette_with_null(X, y, seed=0, n_perm=200, max_n=4000):
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False); X, y = X[idx], y[idx]
    s = silhouette_score(X, y)
    null = np.array([silhouette_score(X, rng.permutation(y)) for _ in range(n_perm)])
    return float(s), float(null.mean()), float(null.std()), float((null >= s).mean())


def knn_purity(X, y, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    y = np.asarray(y)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "corpus_all.parquet"))
    ap.add_argument("--store-root", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "emb_store"))
    ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--site", default="mean"); ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--pca", type=int, default=256, help="PCA dims before probes/geometry (0 = none)")
    ap.add_argument("--out-dir", default=os.path.join(common.REPO, "chrono", "reports", "ssl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    c_all = pd.read_parquet(args.corpus)
    c = c_all[c_all["split"] != "dated"].reset_index(drop=True)   # dated docs keep their own protocol
    cd = c_all[c_all["split"] == "dated"].reset_index(drop=True)  # used ONLY as a held-out TEST source below
    store = EmbStore(args.store_root)
    ids = ("ssl::" + c["uid"].astype(str)).tolist()
    X = store.get(args.model, args.layer, args.site, ids).astype(np.float32)
    ids_d = ("ssl::" + cd["uid"].astype(str)).tolist()
    Xd = None
    if len(ids_d) and bool(np.all(store.has(args.model, args.layer, args.site, ids_d))):
        Xd = store.get(args.model, args.layer, args.site, ids_d).astype(np.float32)
    if args.pca and X.shape[1] > args.pca:
        sc0 = StandardScaler().fit(X)
        pca0 = PCA(args.pca, random_state=args.seed).fit(sc0.transform(X))
        X = pca0.transform(sc0.transform(X)).astype(np.float32)
        if Xd is not None:
            Xd = pca0.transform(sc0.transform(Xd)).astype(np.float32)
    tag = f"{args.model}::L{args.layer}::{args.site}"
    rows, lines = [], [f"# S1 representation probes — {tag}", "",
                       f"texts {len(c):,} · PCA {args.pca} · classes need ≥ {args.min_n} docs", ""]

    def add(metric, value, extra):
        rows.append(dict(run_id=f"s1_probe::{tag}", git_sha="", config_sha="", seed=args.seed,
                         split="ssl_cv", metric=metric, value=float(value), n=int(extra.get("n", 0)),
                         extra=json.dumps(extra)))

    lines += ["## Probes (balanced accuracy, 5-fold, tablet level)", "", "| label | classes | n | linear | MLP | chance |", "|---|---|---|---|---|---|"]
    for lab in ("period_norm", "genre_raw", "provenance", "source"):
        y = c[lab].fillna("NA").astype(str); y = y.where(y != "NA", None)
        m = y.notna(); vc = y[m].value_counts(); keep = vc[vc >= args.min_n].index
        m = m & y.isin(keep)
        if m.sum() < 100 or len(keep) < 2:
            continue
        lin = cv_probe(X[m.to_numpy()], y[m].to_numpy(), "linear", args.seed)
        mlp = cv_probe(X[m.to_numpy()], y[m].to_numpy(), "mlp", args.seed)
        lines.append(f"| {lab} | {lin[2]} | {int(m.sum()):,} | {lin[0]:.3f} ± {lin[1]:.3f} | {mlp[0]:.3f} ± {mlp[1]:.3f} | {1/lin[2]:.3f} |")
        add(f"probe_linear_{lab}", lin[0], {"n": int(m.sum()), "classes": lin[2]})
        add(f"probe_mlp_{lab}", mlp[0], {"n": int(m.sum()), "classes": lin[2]})

    lines += ["", "## Period probe WITHIN source (linear)", "", "| source | classes | n | balanced acc | chance |", "|---|---|---|---|---|"]
    for src, g in c.groupby("source"):
        y = g["period_norm"]; m = y.notna(); vc = y[m].value_counts(); keep = vc[vc >= args.min_n].index; m = m & y.isin(keep)
        if len(keep) >= 2 and m.sum() >= 100:
            r = cv_probe(X[g.index[m]], y[m].to_numpy(), "linear", args.seed)
            lines.append(f"| {src} | {r[2]} | {int(m.sum()):,} | {r[0]:.3f} ± {r[1]:.3f} | {1/r[2]:.3f} |")
            add(f"probe_linear_period_within_{src}", r[0], {"n": int(m.sum()), "classes": r[2]})

    # Period is nearly a proxy for source in this corpus (OB = archibab, LB =
    # letters, Hellenistic = oracc ...), so a held-out source rarely shares its
    # whole label set with the rest. Test only on the classes the probe could
    # learn from the other sources, and add the dated royal inscriptions (S3's
    # labelled set, never seen in SSL training) as one more held-out test source.
    lines += ["", "## Period probe, HELD-OUT source (train on the other non-dated sources, linear)", "",
              "| held out | classes | n test | balanced acc | chance |", "|---|---|---|---|---|"]
    y = c["period_norm"]; m_all = y.notna()
    vc = y[m_all].value_counts(); keep = vc[vc >= args.min_n].index; m_all = m_all & y.isin(keep)
    cases = [(src, (m_all & (c["source"] != src)).to_numpy(), (m_all & (c["source"] == src)).to_numpy(), X, y)
             for src in c.loc[m_all, "source"].unique()]
    if Xd is not None:
        yd = cd["period_norm"]
        cases.append(("orcc (dated)", m_all.to_numpy(), yd.notna().to_numpy(), Xd, yd))
    for src, tr, te, Xte, yte in cases:
        te = te & yte.isin(set(y[tr])).to_numpy()
        if tr.sum() < 100 or te.sum() < 50 or yte[te].nunique() < 2:
            continue
        le = LabelEncoder().fit(y[tr])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), le.transform(y[tr]))
        acc = balanced_accuracy_score(le.transform(yte[te]), clf.predict(sc.transform(Xte[te])))
        k = int(yte[te].nunique())
        lines.append(f"| {src} | {k} | {int(te.sum()):,} | {acc:.3f} | {1/k:.3f} |")
        add(f"probe_linear_period_heldout_{src.split()[0]}", acc, {"n": int(te.sum()), "classes": k})

    lines += ["", "## Geometry (period)", ""]
    m = m_all.to_numpy()
    if m.sum() >= 200:
        s, nm, ns, p = silhouette_with_null(X[m], y[m].to_numpy(), args.seed)
        pur = knn_purity(X[m], y[m].to_numpy())
        lines += [f"silhouette (raw space) {s:+.3f} · permutation null {nm:+.3f} ± {ns:.3f} · p = {p:.3f}",
                  f"k-NN (k=10) period purity {pur:.3f} · chance ≈ {float((y[m].value_counts(normalize=True)**2).sum()):.3f}"]
        add("silhouette_period", s, {"null_mean": nm, "null_sd": ns, "p": p, "n": int(m.sum())})
        add("knn10_purity_period", pur, {"n": int(m.sum())})
        try:
            import umap
            U = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(X[m])
            su, num, nus, pu = silhouette_with_null(U, y[m].to_numpy(), args.seed, n_perm=100)
            lines.append(f"silhouette (UMAP-2d) {su:+.3f} · null {num:+.3f} ± {nus:.3f} · p = {pu:.3f}")
            add("silhouette_period_umap", su, {"null_mean": num, "null_sd": nus, "p": pu})
            os.makedirs(args.out_dir, exist_ok=True)
            pd.DataFrame({"fragment_id": c.loc[m, "fragment_id"].to_numpy(), "u1": U[:, 0], "u2": U[:, 1],
                          "period": y[m].to_numpy(), "source": c.loc[m, "source"].to_numpy()}) \
              .to_parquet(os.path.join(args.out_dir, f"umap_{_safe(args.model)}_L{args.layer}_{args.site}.parquet"), index=False)
        except ImportError:
            lines.append("(umap-learn not installed: UMAP skipped)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"S1_{_safe(args.model)}_L{args.layer}_{args.site}.md")
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    common.append_results(rows)


if __name__ == "__main__":
    main()
