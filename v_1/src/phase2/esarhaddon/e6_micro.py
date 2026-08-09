"""E6 — the Esarhaddon micro-study: the only identity-free document-time test.

WHY HIM. Year is a function of ruler everywhere in this corpus except inside
Esarhaddon: 176 fragments spanning 11 distinct years (681-669 BCE). Inside his
fragments, ruler identity is CONSTANT — so any year signal a probe finds cannot
be an identity lookup, by construction. This is the one place "document-level
time free of identity" is even defined, and it dodges the ICC=1 degeneracy that
made the naive erasure test vacuous.

THE HONEST FLOOR, stated up front: an 12-year window with fragment-year labels
that may themselves derive from textual formulas. Detection is hard here; a null
is compatible with "signal below the floor", and only a positive is decisive.

Three read-outs per arm (all inside the 176 fragments):
  * ridge year probe, 5-fold out-of-fold Spearman, layer swept (stride 2), best
    layer taken at face value and FLAGGED (selection optimism, same convention
    as everywhere else);
  * within-ruler pairwise ordering ("which fragment is earlier"), fragment-level
    folds — both fragments of a test pair unseen in training;
  * permutation p for both: shuffle the year labels inside Esarhaddon (B=1000
    for the probe at the best layer, B=200 with refit for pairwise).

    python e6_micro.py --method tfidf_char --variant akk_maximal
    python e6_micro.py --method olmo2_7b   --variant akk_maximal

Writes results/{method}.{variant}.json. Activations arms need the cluster npz.
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

RESULTS = os.path.join(_HERE, "results")
RULER = "Esarhaddon"
ALPHAS = np.logspace(-1, 5, 13)


def oof_spearman(X, y, seed=0):
    """5-fold out-of-fold ridge predictions -> one pooled Spearman."""
    pred = np.zeros_like(y, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        mu, sd = y[tr].mean(), y[tr].std() + 1e-9
        r = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), (y[tr] - mu) / sd)
        pred[te] = r.predict(sc.transform(X[te])) * sd + mu
    return float(stats.spearmanr(pred, y).correlation), pred


def pairwise_cv(X, df_es, n_pairs, draws, seed):
    """Within-ruler pairwise accuracy, both-fragments-held-out folds."""
    accs = []
    pos2row = {p: i for i, p in enumerate(df_es.pos.values)}
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        pairs = P.draw_within_ruler(df_es, RULER, n_pairs, rng)
        frag_fold = {p: i % 5 for i, p in
                     enumerate(rng.permutation(df_es.pos.values))}
        fa = pairs.pos_a.map(frag_fold).values
        fb = pairs.pos_b.map(frag_fold).values
        correct = []
        for f in range(5):
            tr = pairs[(fa != f) & (fb != f)]
            te = pairs[(fa == f) & (fb == f)]
            if len(te) < 5 or len(tr) < 30:
                continue
            rows = lambda s: np.array([pos2row[p] for p in s])
            sc = StandardScaler().fit(X[np.unique(np.concatenate(
                [rows(tr.pos_a), rows(tr.pos_b)]))])
            Xd = sc.transform(X[rows(tr.pos_a)]) - sc.transform(X[rows(tr.pos_b)])
            clf = LogisticRegression(max_iter=2000, fit_intercept=False)
            clf.fit(Xd, tr.label.values)
            Xt = sc.transform(X[rows(te.pos_a)]) - sc.transform(X[rows(te.pos_b)])
            correct.extend(((clf.decision_function(Xt) > 0).astype(int)
                            == te.label.values).tolist())
        if correct:
            accs.append(float(np.mean(correct)))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal", choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean")
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--n-perm-probe", type=int, default=1000)
    ap.add_argument("--n-perm-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    wc_meta = pd.read_parquet(os.path.join(
        os.path.dirname(os.path.abspath(PP.__file__)), "..", "..", "..",
        "data/evaluation/corpora/orcc_corpus.parquet"
    ))[["fragment_id", "word_count"]]
    df = df.merge(wc_meta, on="fragment_id", how="left")
    es = df[df.ruler == RULER].reset_index(drop=True)
    y = es.year.values.astype(float)
    logwc = np.log1p(es.word_count.fillna(0).values.astype(float))
    # THE CONFOUND THIS SUBSET CARRIES: year correlates with length inside
    # Esarhaddon (rho ~ .38 — mid-reign prisms are huge, reign-edge fragments
    # tiny). Every read-out below therefore ships with a length control.
    rho_yl = float(stats.spearmanr(y, logwc).correlation)
    print(f"[data] {RULER}: {len(es)} fragments, {es.year.nunique()} years "
          f"({int(y.min())}-{int(y.max())}) | rho(year, log-length)={rho_yl:+.3f}",
          flush=True)

    # feature space: one matrix per layer (activations) or one tfidf matrix
    if args.method == "tfidf_char":
        texts = es[PP.TEXT_COL[args.variant]].fillna("").astype(str).values
        # a single in-subset vectorizer refit per fold is what pairwise_cv's
        # scaler slot cannot do for sparse — so for the floor we use SVD-dense
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=120000, sublinear_tf=True)
        Xs = v.fit_transform(texts)
        k = min(256, Xs.shape[1] - 1, len(es) - 1)
        layers = {-1: TruncatedSVD(k, random_state=0).fit_transform(Xs)
                  .astype(np.float32)}
        print(f"[tfidf] SVD-{k} dense floor (vectorizer fitted in-subset — "
              "flagged in output)", flush=True)
    else:
        full = PP.load_act_layers(args.method, args.variant, args.site, stride=2)
        rows = es.pos.values
        layers = {L: X[rows] for L, X in full.items()}

    # --- ridge probe, layer swept ---
    best = (None, -np.inf, None)
    for L, X in layers.items():
        rho, _ = oof_spearman(X, y)
        if rho > best[1]:
            best = (L, rho, X)
    bl, brho, bX = best
    # THE NULL MUST INCLUDE THE SELECTION STEP: the observed statistic is a
    # max over layers, so each permutation also takes its max over layers —
    # a fixed-layer null would be anti-conservative exactly where it matters.
    rng = np.random.default_rng(args.seed)
    n_perm = max(30, args.n_perm_probe // 16)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = max(oof_spearman(X, yp, seed=1)[0]
                      for X in layers.values())
        if (i + 1) % 10 == 0:
            print(f"    probe perm {i + 1}/{n_perm}", flush=True)
    p_probe = float((1 + (null >= brho).sum()) / (n_perm + 1))
    print(f"[probe] best layer {bl}: rho={brho:+.3f}  p={p_probe:.3f} "
          f"(max-over-layers null, mean {null.mean():+.3f}, {n_perm} perms)",
          flush=True)

    # length controls for the probe read-out
    _, bpred = oof_spearman(bX, y)
    len_baseline = oof_spearman(logwc[:, None], y)[0]
    from sklearn.linear_model import LinearRegression
    ry = y - LinearRegression().fit(logwc[:, None], y).predict(logwc[:, None])
    rp = bpred - LinearRegression().fit(logwc[:, None], bpred
                                        ).predict(logwc[:, None])
    partial_rho = float(stats.spearmanr(rp, ry).correlation)
    print(f"[length] rho(year,len)={rho_yl:+.3f} | length-only baseline "
          f"rho={len_baseline:+.3f} | probe PARTIAL rho (length out) = "
          f"{partial_rho:+.3f}", flush=True)

    # --- within-ruler pairwise ---
    acc, sd, nd = pairwise_cv(bX, es, args.n_pairs, args.draws, args.seed)
    nullp = []
    for i in range(args.n_perm_pairs // 4):         # 50 refit perms (p floor .02)
        rngp = np.random.default_rng(9000 + i)
        es_p = es.copy()
        es_p["year"] = rngp.permutation(es_p.year.values)
        a, _, _ = pairwise_cv(bX, es_p, args.n_pairs, 2, args.seed + 500 + i)
        nullp.append(a)
    p_pairs = float((1 + sum(a >= acc for a in nullp)) / (len(nullp) + 1))
    print(f"[pairs] acc={acc:.3f}±{sd:.3f} ({nd} draws)  p={p_pairs:.3f} "
          f"(null mean {np.mean(nullp):.3f}, {len(nullp)} perms)", flush=True)

    out = {"method": args.method, "variant": args.variant, "site": args.site,
           "ruler": RULER, "n_fragments": int(len(es)),
           "n_years": int(es.year.nunique()),
           "probe": {"best_layer": bl, "oof_spearman": brho, "p": p_probe,
                     "null": "max-over-layers (selection included)"},
           "length_control": {"rho_year_length": rho_yl,
                              "length_only_baseline_rho": len_baseline,
                              "probe_partial_rho_length_out": partial_rho},
           "pairwise_layer_from_probe_selection": True,
           "pairwise": {"acc": acc, "sd": sd, "draws": nd, "p": p_pairs},
           "tfidf_vectorizer_in_subset": args.method == "tfidf_char"}
    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"{args.method}.{args.variant}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
