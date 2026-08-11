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


def build_Z(d, concept="all"):
    """Confounder matrix. GENRE IS DELIBERATELY ABSENT — every dated fragment
    in this corpus is genre='Royal Inscription' (checked), so genre has zero
    variance and is not a confounder here at all.

    F28 ladder (--concept): erase ONE variable at a time, so the drop in the
    E1 read-out attributes the document signal variable by variable.
      provenance  find-spot one-hot (top-20 + other)
      length      log word count + 5 quantile bins (basis expansion)
      ruler       ruler-identity one-hot (NOTE the ICC=1 degeneracy: ruler
                  determines year almost 1:1 here, so this rung cannot
                  distinguish "signal is ruler identity" from "signal is
                  year"; read it as the joint upper rung of the era ladder)
      period      the catalogue period column (Neo-Assyrian, Neo-Babylonian,
                  Middle Babylonian, Hellenistic, Achaemenid)
      subgenre    the catalogue sub_genre column = the OBJECT TYPE the text
                  is inscribed on (prism, cylinder, slab, brick, ...);
                  top-20 + other
      year10      year-decile one-hot — coarse era bins; the POSITIVE
                  CONTROL: erasing it must crush any genuinely
                  chronological read-out
      all         provenance + length (the original E4 combination)"""
    top = d.provenance.fillna("unk").value_counts().head(20).index
    prov = pd.get_dummies(d.provenance.fillna("unk").where(
        d.provenance.fillna("unk").isin(top), "other"), dtype=float).values
    wc = np.log1p(d.word_count.fillna(0).astype(float))
    qbins = pd.get_dummies(pd.qcut(wc, 5, duplicates="drop"), dtype=float)
    length = np.hstack([wc.values[:, None], qbins.values])
    if concept == "provenance":
        Z = prov
    elif concept == "length":
        Z = length
    elif concept == "ruler":
        Z = pd.get_dummies(d.ruler, dtype=float).values
    elif concept == "period":
        Z = pd.get_dummies(d.period.fillna("unk"), dtype=float).values
    elif concept == "subgenre":
        sg = d.sub_genre.fillna("unk")
        top = sg.value_counts().head(20).index
        Z = pd.get_dummies(sg.where(sg.isin(top), "other"),
                           dtype=float).values
    elif concept == "year10":
        Z = pd.get_dummies(pd.qcut(d.year.astype(float), 10,
                                   duplicates="drop"), dtype=float).values
    else:                                    # "all" = provenance + length
        Z = np.hstack([prov, length])
    return Z.astype(np.float64)


def concept_series(d, concept):
    """The categorical the post-erasure check probe must fail to read."""
    if concept == "ruler":
        return d.ruler.astype(str)
    if concept == "period":
        return d.period.fillna("unk").astype(str)
    if concept == "subgenre":
        return d.sub_genre.fillna("unk").astype(str)
    if concept == "length":
        wc = np.log1p(d.word_count.fillna(0).astype(float))
        return pd.Series(pd.qcut(wc, 5, duplicates="drop")).astype(str)
    if concept == "year10":
        return pd.Series(pd.qcut(d.year.astype(float), 10,
                                 duplicates="drop")).astype(str)
    return d.provenance                      # provenance / all


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
    ap.add_argument("--concept", default="all",
                    choices=["all", "provenance", "length", "ruler",
                             "period", "subgenre", "year10"],
                    help="F28 ladder: erase a single variable at a time")
    args = ap.parse_args()

    df = P.load_eligible()
    # provenance is already on the eligible frame; merge only what is missing
    meta = pd.read_parquet(os.path.join(
        os.path.dirname(_WM), "..", "data/evaluation/corpora/orcc_corpus.parquet"
    ))[["fragment_id", "genre", "word_count", "period", "sub_genre"]]
    df = df.merge(meta, on="fragment_id", how="left").reset_index(drop=True)
    Z = build_Z(df, args.concept)
    y = df.year.values.astype(float)
    print(f"[data] {len(df)} frags, concept={args.concept}, "
          f"Z dim {Z.shape[1]}", flush=True)

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
           "concept": args.concept, "z_dim": int(Z.shape[1]),
           "m": args.m, "draws": args.draws}
    for tag, erase in (("raw", False), ("erased", True)):
        mac, sd = pairwise_macro(df, X, Z, args.m, args.draws,
                                 args.seed, erase)
        rho = grouped_ridge(X, y, df.ruler.values,
                            erase_Z=Z if erase else None)
        out[tag] = {"pairwise_macro": mac, "pairwise_sd": sd,
                    "grouped_ridge_spearman": rho}
        print(f"[{tag}] pairwise={mac:.3f}±{sd:.3f} ridge_rho={rho:+.3f}",
              flush=True)
    # did the erasure actually remove the concept? (manipulation check)
    e_all = fit_eraser(X, Z)
    Xe = e_all(X)
    chk = concept_series(df, args.concept)
    out["concept_probe_acc_before"] = provenance_probe_acc(X, chk)
    out["concept_probe_acc_after"] = provenance_probe_acc(Xe, chk)
    if args.concept == "all":        # backward-compatible keys
        out["prov_probe_acc_before"] = out["concept_probe_acc_before"]
        out["prov_probe_acc_after"] = out["concept_probe_acc_after"]
    out["genre_constant"] = True     # every dated fragment is Royal Inscription
    print(f"[check] {args.concept} probe "
          f"{out['concept_probe_acc_before']:.3f} -> "
          f"{out['concept_probe_acc_after']:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    name = (f"{args.method}.{args.variant}.json" if args.concept == "all"
            else f"ladder.{args.concept}.{args.method}.{args.variant}.json")
    pth = os.path.join(RESULTS, name)
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
