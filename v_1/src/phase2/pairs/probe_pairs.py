"""E1 representation-side probe: can a linear scorer order fragment PAIRS by time?

WHAT IS LEARNED. A direction w such that w.(x_a - x_b) > 0 predicts "a earlier".
A linear pairwise-logistic scorer is a Bradley-Terry model with linear features:
it learns a TIME DIRECTION from relative order only, never touching absolute year
values — which is the point, because the absolute labels are the thing the thesis
does not want to lean on. The fitted w is saved so it can be compared (cosine)
against the frozen cell-A name direction in E3.

PROTOCOL (mirrors the house mc_group discipline, translated to pairs):
  * balanced draws from pairs_data (quota m per ruler-pair, weight 1/m_ij);
  * ruler folds reshuffled per draw; a pair TRAINS only when both rulers are in
    train folds and TESTS only when both are in the test fold — the pairwise
    analog of GroupKFold-by-ruler;
  * scaler fitted on train-fragment vectors only; for tfidf, the vectorizer too;
  * metrics macro-averaged over ruler-pairs, then mean+-sd over draws;
  * layer chosen by the same macro metric on a cheaper selection pass
    (--draws-select), then the full Monte-Carlo runs at that layer. Same mild
    optimism as the deck's holdout-best-layer convention; flagged in the JSON.

ARMS. --method is an activation dir under world_models/akkadian/activations
(olmo2_7b, llama2_7b_random, qwen3_8b, ...) or the text floor `tfidf_char`
(char_wb 2-5 gram TF-IDF, the vectorizer settings of tfidf_akk.py).

    python probe_pairs.py --method tfidf_char --variant akk_maximal
    python probe_pairs.py --method olmo2_7b  --variant akk_maximal --site mean

Writes results/probes/{method}.{variant}.{site}.json
   and results/directions/{method}.{variant}.{site}.layer{L}.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pairs_data as P                                  # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import probing                              # noqa: E402

ACTS = os.path.join(_WM, "akkadian", "activations")
RESULTS = os.path.join(_HERE, "results")
DYEAR_BINS = [0, 25, 75, 200, np.inf]
TEXT_COL = {"akk_maximal": "text_akk", "eng_maximal": "text_eng",
            "eng_tier0": "text_eng_tier0"}


# ---------------------------------------------------------------- feature spaces
def load_act_layers(method, variant, site, stride):
    d = os.path.join(ACTS, method, variant)
    files = sorted(glob.glob(os.path.join(d, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    if not files:
        sys.exit(f"no activations under {d} for site={site} — run extract_akk "
                 f"for {method} first (cluster; npz are not in git).")
    out = {}
    for pth in files[::stride] if stride > 1 else files:
        li = int(re.search(r"layer(\d+)\.npz$", pth).group(1))
        X = np.load(pth)["acts"].astype(np.float32)
        X, bad = probing.sanitize(X)
        if bad <= 0.01:
            out[li] = X
        else:
            print(f"[warn] layer {li}: {bad:.1%} non-finite, skipped", flush=True)
    return out


def make_tfidf(texts_train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                        max_features=120000, lowercase=True, sublinear_tf=True)
    v.fit(texts_train)
    return v


# ---------------------------------------------------------------- one MC pass
def run_mc(df, get_feats, m, draws, seed0, rp, sparse=False):
    """get_feats(train_pos) -> (transform(pos_array) -> matrix). Refit per fold so
    nothing about the test rulers leaks into scaling/vocabulary."""
    per_draw = []
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        pairs = P.draw_pairs(df, m, rng, rp)
        folds = P.ruler_folds(sorted(df.ruler.unique()), rng)
        fa = pairs.ruler_a.map(folds).values
        fb = pairs.ruler_b.map(folds).values
        rows = []
        for f in sorted(set(folds.values())):
            tr = pairs[(fa != f) & (fb != f)]
            te = pairs[(fa == f) & (fb == f)]
            if len(te) < 5 or len(tr) < 50:
                continue
            tr_pos = np.unique(np.concatenate([tr.pos_a.values, tr.pos_b.values]))
            transform = get_feats(tr_pos)
            Xtr = transform(tr.pos_a.values) - transform(tr.pos_b.values)
            Xte = transform(te.pos_a.values) - transform(te.pos_b.values)
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     solver="liblinear" if sparse else "lbfgs")
            clf.fit(Xtr, tr.label.values, sample_weight=tr.weight.values)
            s = clf.decision_function(Xte)
            rows.append(pd.DataFrame({
                "score": s, "label": te.label.values, "dyear": te.dyear.values,
                "rp": [f"{min(a, b)}|{max(a, b)}"
                       for a, b in zip(te.ruler_a, te.ruler_b)]}))
        if not rows:
            continue
        t = pd.concat(rows, ignore_index=True)
        t["correct"] = ((t.score > 0).astype(int) == t.label).astype(float)
        rec = {
            "macro_acc": float(t.groupby("rp")["correct"].mean().mean()),
            "micro_acc": float(t.correct.mean()),
            "auc": float(roc_auc_score(t.label, t.score))
            if t.label.nunique() == 2 else float("nan"),
            "n_test_pairs": int(len(t)),
            "n_ruler_pairs": int(t.rp.nunique()),
        }
        cut = pd.cut(t.dyear, DYEAR_BINS, right=False)
        rec["acc_by_dyear"] = {str(iv): float(g.correct.mean())
                               for iv, g in t.groupby(cut, observed=True)
                               if len(g) >= 20}
        per_draw.append(rec)
    return per_draw


def summarize(per_draw):
    if not per_draw:
        return {"skipped": True}
    out = {}
    for k in ("macro_acc", "micro_acc", "auc"):
        v = np.array([r[k] for r in per_draw], float)
        out[f"{k}_mean"], out[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    out["n_draws"] = len(per_draw)
    out["n_test_pairs_mean"] = float(np.mean([r["n_test_pairs"] for r in per_draw]))
    out["n_ruler_pairs_mean"] = float(np.mean([r["n_ruler_pairs"] for r in per_draw]))
    bins = {}
    for r in per_draw:
        for b, v in r["acc_by_dyear"].items():
            bins.setdefault(b, []).append(v)
    out["acc_by_dyear"] = {b: {"mean": float(np.mean(v)), "n_draws": len(v)}
                           for b, v in sorted(bins.items())}
    return out


# ---------------------------------------------------------------- direction save
def fit_direction(df, get_feats, m, seed, rp):
    """One full-data fit (all rulers) purely to EXPORT the learned direction for
    the E3 cosine comparison. Never used for any score reported here."""
    rng = np.random.default_rng(seed)
    pairs = P.draw_pairs(df, m, rng, rp)
    transform = get_feats(df.pos.values)
    Xd = transform(pairs.pos_a.values) - transform(pairs.pos_b.values)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xd, pairs.label.values, sample_weight=pairs.weight.values)
    return clf.coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal", choices=list(TEXT_COL))
    ap.add_argument("--site", default="mean")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=P.N_DRAWS_DEFAULT)
    ap.add_argument("--draws-select", type=int, default=6,
                    help="cheaper MC used only to pick the layer")
    ap.add_argument("--layer-stride", type=int, default=2,
                    help="thin the layer sweep during selection (full MC runs at "
                         "one layer anyway)")
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    rp = P.eligible_ruler_pairs(df)
    print(f"[data] {len(df)} fragments, {df.ruler.nunique()} rulers, "
          f"{len(rp)} eligible ruler-pairs, m={args.m}", flush=True)

    if args.method == "tfidf_char":
        texts = df[TEXT_COL[args.variant]].fillna("").astype(str).values
        pos2row = {p: i for i, p in enumerate(df.pos.values)}

        def get_feats(tr_pos):
            v = make_tfidf([texts[pos2row[p]] for p in tr_pos])
            def transform(pos_arr):
                return v.transform([texts[pos2row[p]] for p in pos_arr])
            return transform
        per_layer = {-1: summarize(run_mc(df, get_feats, args.m,
                                          args.draws, args.seed, rp, sparse=True))}
        best_layer, w = -1, None
    else:
        layers = load_act_layers(args.method, args.variant, args.site,
                                 args.layer_stride)
        print(f"[acts] {len(layers)} layers loaded", flush=True)

        def feats_for(X):
            def get_feats(tr_pos):
                sc = StandardScaler().fit(X[tr_pos])
                def transform(pos_arr):
                    return sc.transform(X[pos_arr])
                return transform
            return get_feats

        sel = {}
        for li, X in layers.items():
            sel[li] = summarize(run_mc(df, feats_for(X), args.m,
                                       args.draws_select, args.seed, rp))
            print(f"  select layer {li}: macro_acc="
                  f"{sel[li].get('macro_acc_mean', float('nan')):.3f}", flush=True)
        best_layer = max(sel, key=lambda L: sel[L].get("macro_acc_mean", -np.inf))
        per_layer = {best_layer: summarize(run_mc(
            df, feats_for(layers[best_layer]), args.m, args.draws,
            args.seed + 1000, rp))}
        per_layer[best_layer]["selection"] = {str(k): v for k, v in sel.items()}
        w = fit_direction(df, feats_for(layers[best_layer]), args.m, args.seed, rp)

    full = per_layer[best_layer]
    out = {"method": args.method, "variant": args.variant, "site": args.site,
           "m": args.m, "draws": args.draws, "n_fragments": int(len(df)),
           "n_rulers": int(df.ruler.nunique()), "n_ruler_pairs": len(rp),
           "best_layer": best_layer,
           "layer_selected_on_same_protocol": args.method != "tfidf_char",
           "full": full}
    os.makedirs(os.path.join(RESULTS, "probes"), exist_ok=True)
    pth = os.path.join(RESULTS, "probes",
                       f"{args.method}.{args.variant}.{args.site}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    if w is not None:
        os.makedirs(os.path.join(RESULTS, "directions"), exist_ok=True)
        np.savez_compressed(
            os.path.join(RESULTS, "directions",
                         f"{args.method}.{args.variant}.{args.site}"
                         f".layer{best_layer}.npz"),
            w=w.astype(np.float32))
    print(f"[done] {args.method}/{args.variant}/{args.site} layer {best_layer}: "
          f"macro_acc={full.get('macro_acc_mean', float('nan')):.3f}"
          f"±{full.get('macro_acc_std', float('nan')):.3f} "
          f"auc={full.get('auc_mean', float('nan')):.3f} -> {pth}", flush=True)


if __name__ == "__main__":
    main()
