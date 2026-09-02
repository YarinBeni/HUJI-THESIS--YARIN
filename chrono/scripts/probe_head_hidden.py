"""probe_head_hidden.py — nonlinear-recovery check for the P1 head ladder.

LEACE guarantees only that NO LINEAR probe can read the erased concept from the
features. The Chrono-Barlow head is nonlinear, so its post-erasure dating
accuracy could in principle come from re-deriving provenance nonlinearly.
Test: per fold, rebuild the erased features exactly as train_cjb did (eraser
fitted on the train docs' orig views), load the saved head, take its hidden
layer h = mlp(X_erased) on the fold's train docs, and ask a linear AND a
nonlinear (MLP) probe to read provenance (classes >= 10 docs) on a within-
train 80/20 split. Reference points: the same probes on raw X (before
erasure) and on X_erased itself.

If provenance is at chance from h -> the head's post-erasure rho is
non-site chronology. If it is well above chance -> the margin may be
site-driven after all.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
from chrono.models.heads import AdapterHead                     # noqa: E402
from chrono.eval.erasure import LeaceEraser, concept_matrix     # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402
from sklearn.metrics import balanced_accuracy_score             # noqa: E402


def probe(Xa, za, Xb, zb, kind, seed=0):
    sc = StandardScaler().fit(Xa)
    if kind == "linear":
        clf = LogisticRegression(max_iter=3000, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=seed,
                            early_stopping=True)
    clf.fit(sc.transform(Xa), za)
    return float(balanced_accuracy_score(zb, clf.predict(sc.transform(Xb))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--heads-dir", default="chrono/reports/tier0/heads")
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--concept", default="provenance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config)); feats = cfg["features"]; run = cfg["run_name"]
    langs = cfg["views"].get("langs") or ["akk", "eng"]
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet")).reset_index(drop=True)
    gkf = json.load(open(os.path.join(args.art, "splits", "gkf_ruler.json")))
    store = EmbStore(os.path.join(args.art, "emb_store"))
    ids = corpus["doc_id"].astype(str).tolist()
    X = np.mean([store.get(feats["model"], feats["layer"], feats["site"],
                           [f"{d}::{lg}::orig" for d in ids]) for lg in langs], axis=0).astype(np.float32)
    Z, z = concept_matrix(corpus, args.concept)
    vc = pd.Series(z).value_counts(); freq = set(vc[vc >= 10].index)
    # integer labels: sklearn's MLPClassifier early-stopping scorer calls
    # np.isnan on predictions and chokes on string classes (job 018)
    from sklearn.preprocessing import LabelEncoder
    zi = LabelEncoder().fit(z).transform(z)
    rows = []
    for k, fold in enumerate(gkf["folds"]):
        tr = corpus["doc_id"].isin(set(fold["train"])).to_numpy()
        er = LeaceEraser().fit(X[tr], Z[tr]); Xe = er(X)
        pt = os.path.join(args.heads_dir, f"{run}-s0-f{k}.pt")
        if not os.path.exists(pt):
            print(f"[probe] missing head {pt}; skipping fold {k}"); continue
        head = AdapterHead(d_in=X.shape[1]); head.load_state_dict(torch.load(pt, map_location="cpu")); head.eval()
        with torch.no_grad():
            H = head.mlp(torch.as_tensor(Xe)).numpy()
        rows_tr = np.flatnonzero(tr & np.isin(z, list(freq)))
        rng = np.random.default_rng(k); perm = rng.permutation(rows_tr); cut = int(0.8 * len(perm))
        a, b = perm[:cut], perm[cut:]
        if len(set(zi[a])) < 2 or not set(zi[b]) <= set(zi[a]):
            continue
        for name, F in (("X raw", X), ("X erased", Xe), ("head hidden h(X erased)", H)):
            for kind in ("linear", "mlp"):
                rows.append(dict(fold=k, features=name, probe=kind,
                                 bal_acc=probe(F[a], zi[a], F[b], zi[b], kind)))
        print(f"[probe] fold {k} done", flush=True)
    tab = pd.DataFrame(rows)
    agg = tab.groupby(["features", "probe"], sort=False)["bal_acc"].agg(["mean", "std"]).reset_index()
    chance = 1.0 / len(freq)
    out = args.out or f"chrono/reports/tier0/ladder/nonlinear_recovery_{run}.md"
    with open(out, "w") as f:
        f.write(f"# Nonlinear-recovery check — {run} — erased concept `{args.concept}` "
                f"({len(freq)} classes ≥10 docs, chance {chance:.2f})\n\n| features | probe | balanced acc (mean ± sd over folds) |\n|---|---|---|\n")
        for r in agg.itertuples():
            f.write(f"| {r.features} | {r.probe} | {r.mean:.3f} ± {r.std:.3f} |\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
