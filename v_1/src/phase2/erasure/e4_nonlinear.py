"""F27 — nonlinear probes (the E4 leftover, and a standing hidden assumption).

Every probe in the program has been linear (ridge / logistic / PLS), and every
results section carries the caveat "a nonlinear representation of document
time would not be caught". This closes it: two standard nonlinear heads —
RBF kernel ridge (median-heuristic gamma grid) and a small MLP — trained
under the SAME leakage discipline (GroupKFold by ruler, score pairs only
where BOTH rulers are in the held-out fold), against the same floor and twin.

Protocol per arm:
  * X = mean-pooled activations at the arm's selected layer (the layer comes
    from the existing F1 probe file — fixed, not re-searched); tfidf goes
    through SVD-512 as in E4 so the heads see a dense space;
  * standardize -> PCA-256 (fit on train folds only) for the nonlinear heads;
  * OOF year predictions from GroupKFold(5) by ruler; report
      - pooled OOF Spearman(y_hat, year),
      - pairwise macro over ruler-pairs with both rulers held out, decided by
        sign(y_hat_a - y_hat_b), quota m per ruler pair (E1's balancing);
  * heads: ridge (reference), kernel-RBF, MLP(256,128) with early stopping.

PRE-REGISTERED RULES: (1) if for every model arm the best nonlinear head
beats neither its own linear head by >.02 macro nor the tfidf floor, the
claim "no document-time signal, linear OR nonlinear" is sealed and the
caveat is retired. (2) A model arm whose nonlinear head beats the floor AND
its twin by >2 pooled sd is a new lead — report, do not celebrate (one
comparison per arm, no head-shopping: the MLP and kernel are BOTH reported).

    python e4_nonlinear.py --method olmo2_7b --variant eng_tier0

Writes erasure/results/nl.{method}.{variant}.json. CPU is enough.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

RESULTS = os.path.join(_HERE, "results")


def heads(seed):
    return {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-1, 5, 13)),
        "krr_rbf": None,      # built per-fold (needs the median heuristic)
        "mlp": lambda: MLPRegressor(hidden_layer_sizes=(256, 128),
                                    early_stopping=True, max_iter=500,
                                    random_state=seed),
    }


def krr_median(Xtr, ytr, seed):
    """RBF kernel ridge; gamma from the median pairwise-distance heuristic
    x {0.5, 1, 2}, alpha in a small grid, chosen by an inner 3-fold CV on the
    TRAIN side only."""
    from sklearn.model_selection import GridSearchCV, KFold
    sub = Xtr[np.random.default_rng(seed).choice(
        len(Xtr), min(400, len(Xtr)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    g0 = 1.0 / max(med, 1e-8)
    gs = GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"gamma": [g0 * f for f in (0.5, 1.0, 2.0)],
         "alpha": [0.1, 1.0, 10.0]},
        cv=KFold(3, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_


def oof_predictions(X, y, rulers, head_name, seed):
    """GroupKFold(5)-by-ruler OOF predictions with per-fold scaler+PCA."""
    yhat = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, rulers)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if head_name != "ridge":
            pca = PCA(n_components=min(256, Xtr.shape[1], len(tr) - 1),
                      random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        # y standardized per train fold: MLP/adam needs a sane target scale,
        # and every read-out below is ordinal (Spearman / pair sign), so the
        # monotone transform is free
        mu, sd = y[tr].mean(), y[tr].std() + 1e-8
        ytr = (y[tr] - mu) / sd
        if head_name == "krr_rbf":
            mdl = krr_median(Xtr, ytr, seed)
        else:
            mdl = heads(seed)[head_name]()
            mdl.fit(Xtr, ytr)
        yhat[te] = mdl.predict(Xte)
        fold_of[te] = k
    return yhat, fold_of


def pairwise_macro_from_scores(df, yhat, fold_of, m, seed):
    """Macro accuracy over ruler pairs where BOTH rulers share a held-out
    fold (so neither ruler was in that model's training side)."""
    rng = np.random.default_rng(seed)
    accs = []
    d = df.assign(score=yhat, fold=fold_of)
    for f in sorted(set(fold_of)):
        dd = d[d.fold == f]
        rulers = dd.ruler.unique()
        for i in range(len(rulers)):
            for j in range(i + 1, len(rulers)):
                a = dd[dd.ruler == rulers[i]]
                b = dd[dd.ruler == rulers[j]]
                ia = rng.choice(len(a), min(m, len(a)), replace=False)
                ib = rng.choice(len(b), min(m, len(b)), replace=False)
                # per-FRAGMENT years (Esarhaddon's year varies within reign)
                ya = a.year.values[ia][:, None]
                yb = b.year.values[ib][None, :]
                sa = a.score.values[ia][:, None]
                sb = b.score.values[ib][None, :]
                valid = ya != yb
                if not valid.any():
                    continue
                correct = ((sa < sb) == (ya < yb))[valid]
                accs.append(float(correct.mean()))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="eng_tier0",
                    choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    df = P.load_eligible()
    y = df.year.values.astype(float)
    if args.method == "tfidf_char":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df[PP.TEXT_COL[args.variant]].fillna("").astype(str).values
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=120000, sublinear_tf=True)
        X = TruncatedSVD(512, random_state=0).fit_transform(
            v.fit_transform(texts)).astype(np.float32)
        bl = -1
    else:
        f1 = os.path.join(_PAIRS, "results", "probes",
                          f"{args.method}.{args.variant}.{args.site}.json")
        bl = json.load(open(f1))["best_layer"]
        X = PP.load_act_layers(args.method, args.variant, args.site,
                               stride=1)[bl][df.pos.values]
    print(f"[data] {args.method}/{args.variant} X={X.shape} layer={bl}",
          flush=True)

    out = {"method": args.method, "variant": args.variant, "layer": int(bl),
           "heads": {}}
    for head in ("ridge", "krr_rbf", "mlp"):
        yhat, fold_of = oof_predictions(X, y, df.ruler.values, head,
                                        args.seed)
        rho = float(stats.spearmanr(yhat, y).correlation)
        mac, sd, npairs = pairwise_macro_from_scores(df, yhat, fold_of,
                                                     args.m, args.seed)
        out["heads"][head] = {"oof_spearman": rho, "pairwise_macro": mac,
                              "pairwise_sd": sd, "n_ruler_pairs": npairs}
        print(f"[{head}] rho={rho:+.3f} macro={mac:.3f}±{sd:.3f} "
              f"({npairs} ruler-pairs)", flush=True)

    pth = os.path.join(RESULTS, f"nl.{args.method}.{args.variant}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
