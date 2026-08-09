"""E4 — confounder erasure: does any chronological signal survive removing
genre, length, and find-spot?

THE WORRY IT ANSWERS. A "time" signal in cell C could really be REGISTER:
building inscriptions cluster in one era, annalistic texts in another, sites
were occupied at different times, and later kings left longer texts. LEACE
(Belrose et al. 2023) removes the linear trace of these confounders from the
representations — fitted on TRAIN fragments only, applied to both sides — and
the question is what survives.

Confounder matrix Z per fragment: genre one-hot + provenance one-hot (top-20 +
other) + log length + 5 length-quantile bins (basis expansion, since LEACE
removes only linear dependence).

Read-outs, before vs after erasure, same protocols as E1/E8:
  * pairwise macro accuracy (both-rulers-held-out, m=21);
  * a GroupKFold-by-ruler ridge year probe (pooled OOF Spearman);
  * the confound-check: can a linear probe still read genre after erasure?

The floor moves consistently: tfidf goes through SVD-512 first (LEACE needs a
dense space), and that reduction is applied before AND after so the comparison
is like-for-like.

    python e4_confounders.py --method olmo2_7b --variant eng_tier0
    python e4_confounders.py --method tfidf_char --variant akk_maximal

Writes results/{method}.{variant}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
RESULTS = os.path.join(_HERE, "results")
ALPHAS = np.logspace(-1, 5, 13)


def build_Z(d):
    """Confounders: provenance + length. GENRE IS DELIBERATELY ABSENT — every
    dated fragment in this corpus is genre='Royal Inscription' (checked), so
    genre has zero variance and is not a confounder here at all. That is worth
    a sentence in the thesis: the register worry reduces to find-spot + length."""
    top = d.provenance.fillna("unk").value_counts().head(20).index
    prov = pd.get_dummies(d.provenance.fillna("unk").where(
        d.provenance.fillna("unk").isin(top), "other"), dtype=float)
    wc = np.log1p(d.word_count.fillna(0).astype(float))
    qbins = pd.get_dummies(pd.qcut(wc, 5, duplicates="drop"), dtype=float)
    Z = np.hstack([prov.values, wc.values[:, None], qbins.values])
    return Z.astype(np.float64)


def fit_eraser(Xtr, Ztr):
    import torch
    from concept_erasure import LeaceEraser
    er = LeaceEraser.fit(torch.from_numpy(Xtr.astype(np.float64)),
                         torch.from_numpy(Ztr))
    return lambda X: er(torch.from_numpy(X.astype(np.float64))).float().numpy()


def grouped_ridge(X, y, rulers, erase_Z=None, seed=0):
    """GroupKFold-by-ruler OOF Spearman; optional per-fold LEACE of Z."""
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=rulers):
        Xtr, Xte = X[tr], X[te]
        if erase_Z is not None:
            e = fit_eraser(Xtr, erase_Z[tr])
            Xtr, Xte = e(Xtr), e(Xte)
        sc = StandardScaler().fit(Xtr)
        mu, sd = y[tr].mean(), y[tr].std() + 1e-9
        r = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), (y[tr] - mu) / sd)
        pred[te] = r.predict(sc.transform(Xte)) * sd + mu
    ok = np.isfinite(pred)
    return float(stats.spearmanr(pred[ok], y[ok]).correlation)


def pairwise_macro(df, X, Z, m, draws, seed, erase):
    """The E1 protocol with optional in-fold erasure."""
    rp = P.eligible_ruler_pairs(df)
    pos2row = {p: i for i, p in enumerate(df.pos.values)}
    per_draw = []
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        pairs = P.draw_pairs(df, m, rng, rp)
        folds = P.ruler_folds(sorted(df.ruler.unique()), rng)
        fa = pairs.ruler_a.map(folds).values
        fb = pairs.ruler_b.map(folds).values
        rows_all = []
        for f in sorted(set(folds.values())):
            tr = pairs[(fa != f) & (fb != f)]
            te = pairs[(fa == f) & (fb == f)]
            if len(te) < 5 or len(tr) < 50:
                continue
            r = lambda s: np.array([pos2row[p] for p in s])
            tr_rows = np.unique(np.concatenate([r(tr.pos_a), r(tr.pos_b)]))
            Xf = X
            if erase:
                e = fit_eraser(X[tr_rows], Z[tr_rows])
                Xf = e(X)
            sc = StandardScaler().fit(Xf[tr_rows])
            Xd = sc.transform(Xf[r(tr.pos_a)]) - sc.transform(Xf[r(tr.pos_b)])
            clf = LogisticRegression(max_iter=2000, fit_intercept=False)
            clf.fit(Xd, tr.label.values, sample_weight=tr.weight.values)
            Xt = sc.transform(Xf[r(te.pos_a)]) - sc.transform(Xf[r(te.pos_b)])
            correct = ((clf.decision_function(Xt) > 0).astype(int)
                       == te.label.values)
            rows_all.append(pd.DataFrame({
                "c": correct.astype(float),
                "rp": [f"{min(a, b)}|{max(a, b)}"
                       for a, b in zip(te.ruler_a, te.ruler_b)]}))
        if rows_all:
            t = pd.concat(rows_all)
            per_draw.append(float(t.groupby("rp").c.mean().mean()))
    return float(np.mean(per_draw)), float(np.std(per_draw))


def provenance_probe_acc(X, prov, seed=0):
    """The erasure check: can a linear probe still read the find-spot?"""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    g = pd.Categorical(prov.fillna("unk")).codes
    ok = pd.Series(g).groupby(g).transform("size").values >= 5
    if pd.Series(g[ok]).nunique() < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, C=0.1))
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X[ok], g[ok], cv=cv).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="eng_tier0", choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    # provenance is already on the eligible frame; merge only what is missing
    meta = pd.read_parquet(os.path.join(
        os.path.dirname(_WM), "..", "data/evaluation/corpora/orcc_corpus.parquet"
    ))[["fragment_id", "genre", "word_count"]]
    df = df.merge(meta, on="fragment_id", how="left").reset_index(drop=True)
    Z = build_Z(df)
    y = df.year.values.astype(float)
    print(f"[data] {len(df)} frags, Z dim {Z.shape[1]}", flush=True)

    if args.method == "tfidf_char":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df[PP.TEXT_COL[args.variant]].fillna("").astype(str).values
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=120000, sublinear_tf=True)
        X = TruncatedSVD(512, random_state=0).fit_transform(
            v.fit_transform(texts)).astype(np.float32)
        bl = -1
        print("[tfidf] SVD-512 dense (vectorizer+SVD fitted on all rows — "
              "flagged; both arms share the bias)", flush=True)
    else:
        f1 = os.path.join(_PAIRS, "results", "probes",
                          f"{args.method}.{args.variant}.{args.site}.json")
        bl = json.load(open(f1))["best_layer"]
        X = PP.load_act_layers(args.method, args.variant, args.site,
                               stride=1)[bl][df.pos.values]

    out = {"method": args.method, "variant": args.variant, "layer": bl,
           "z_dim": int(Z.shape[1]), "m": args.m, "draws": args.draws}
    for tag, erase in (("raw", False), ("erased", True)):
        mac, sd = pairwise_macro(df, X, Z, args.m, args.draws,
                                 args.seed, erase)
        rho = grouped_ridge(X, y, df.ruler.values,
                            erase_Z=Z if erase else None)
        out[tag] = {"pairwise_macro": mac, "pairwise_sd": sd,
                    "grouped_ridge_spearman": rho}
        print(f"[{tag}] pairwise={mac:.3f}±{sd:.3f} ridge_rho={rho:+.3f}",
              flush=True)
    # did the erasure actually remove the confounder?
    e_all = fit_eraser(X, Z)
    out["prov_probe_acc_before"] = provenance_probe_acc(X, df.provenance)
    out["prov_probe_acc_after"] = provenance_probe_acc(e_all(X), df.provenance)
    out["genre_constant"] = True     # every dated fragment is Royal Inscription
    print(f"[check] provenance probe {out['prov_probe_acc_before']:.3f} -> "
          f"{out['prov_probe_acc_after']:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"{args.method}.{args.variant}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
