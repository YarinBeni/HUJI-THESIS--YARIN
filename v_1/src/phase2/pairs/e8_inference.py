"""E8 for E1: cluster-honest inference on the pairwise-chronology results.

E1's mean +- sd is spread over Monte-Carlo draws — resamples of the same corpus,
not independent evidence. The independent units are RULERS (40 of them), and every
pair touches two, so the data are dyadic. Two complementary tests:

1. DYADIC BOOTSTRAP (two-way ruler resampling) for the paired differences
   arm - floor and arm - twin. All arms are scored on IDENTICAL draws and fold
   assignments (shared seeds), so per-ruler-pair accuracies subtract cleanly.
   Each bootstrap replicate resamples the 40 rulers with replacement and weights
   ruler-pair (a, b) by count_a * count_b — the standard bootstrap for dyadic
   data (Snijders & Borgatti 1999). Reported: 95% CI and a two-sided p for the
   macro difference.

2. RULER-LEVEL PERMUTATION with refit: under the null "ruler chronology is
   exchangeable", the ruler-level years are shuffled among rulers, pair labels
   rebuilt, and the probe REFIT from scratch per permutation. The observed macro
   accuracy is ranked against the permutation distribution:
   p = (1 + #{perm >= obs}) / (n_perm + 1). This kills the leak that made the
   'mc' protocol lie: nothing about the true chronology survives into the null.
   (The layer stays fixed at the F1-selected best layer; selection on real labels
   makes this marginally anti-conservative and the JSON says so.)

Esarhaddon's within-ruler year spread collapses under the ruler-level shuffle;
that is the correct null for the claim being tested (BETWEEN-ruler chronology).

    python e8_inference.py --variant eng_tier0 \
        --arms tfidf_char olmo2_7b olmo2_7b_random qwen3_8b random
    python e8_inference.py --variant akk_maximal --arms tfidf_char olmo2_7b

Writes results/inference/{variant}.json (+ per-arm rp tables under
results/inference/rp/). CPU only; activations required for LLM arms (cluster).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pairs_data as P                                  # noqa: E402
import probe_pairs as PP                                # noqa: E402

RESULTS = os.path.join(_HERE, "results")
FLOOR = "tfidf_char"


# ------------------------------------------------------------ shared draw plans
def build_plans(df, m, draws, seed, rp):
    """The SAME pair draws + fold assignments for every arm, so per-ruler-pair
    accuracies are paired across arms and their differences are meaningful."""
    plans = []
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        pairs = P.draw_pairs(df, m, rng, rp)
        folds = P.ruler_folds(sorted(df.ruler.unique()), rng)
        plans.append((pairs, folds))
    return plans


def arm_features(df, method, variant, site, fixed_layer):
    """A get_feats(train_pos) factory for one arm at one fixed layer."""
    if method == FLOOR:
        texts = df[PP.TEXT_COL[variant]].fillna("").astype(str).values
        pos2row = {p: i for i, p in enumerate(df.pos.values)}

        def get_feats(tr_pos):
            v = PP.make_tfidf([texts[pos2row[p]] for p in tr_pos])
            return lambda pos: v.transform([texts[pos2row[p]] for p in pos])
        return get_feats, True
    layers = PP.load_act_layers(method, variant, site, stride=1)
    if fixed_layer not in layers:
        sys.exit(f"{method}: layer {fixed_layer} not on disk (have "
                 f"{sorted(layers)[:5]}...)")
    X = layers[fixed_layer]

    def get_feats(tr_pos):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr_pos])
        return lambda pos: sc.transform(X[pos])
    return get_feats, False


def score_plans(df, get_feats, sparse, plans):
    """Run the E1 protocol on prebuilt plans; return a per-(draw, rp) accuracy
    frame. Identical logic to probe_pairs.run_mc, minus the summary step."""
    rows = []
    for d, (pairs, folds) in enumerate(plans):
        fa = pairs.ruler_a.map(folds).values
        fb = pairs.ruler_b.map(folds).values
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
            correct = ((clf.decision_function(Xte) > 0).astype(int)
                       == te.label.values).astype(float)
            rows.append(pd.DataFrame({
                "draw": d, "correct": correct,
                "ra": np.minimum(te.ruler_a.values, te.ruler_b.values),
                "rb": np.maximum(te.ruler_a.values, te.ruler_b.values)}))
    t = pd.concat(rows, ignore_index=True)
    return (t.groupby(["draw", "ra", "rb"])["correct"].mean()
             .rename("acc").reset_index())


def rp_table(scores):
    """Per-ruler-pair accuracy averaged over the draws that tested it."""
    return scores.groupby(["ra", "rb"])["acc"].mean()


def macro(scores):
    return float(scores.groupby("draw")
                 .apply(lambda g: g.acc.mean(), include_groups=False).mean())


# ------------------------------------------------------------ dyadic bootstrap
def dyadic_boot(diff_by_rp, rulers, n_boot, seed):
    """Two-way ruler resampling of a per-ruler-pair statistic."""
    rng = np.random.default_rng(seed)
    idx = {r: i for i, r in enumerate(rulers)}
    ra = np.array([idx[a] for a, _ in diff_by_rp.index])
    rb = np.array([idx[b] for _, b in diff_by_rp.index])
    vals = diff_by_rp.values
    out = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(rulers), len(rulers)),
                          minlength=len(rulers))
        w = cnt[ra] * cnt[rb]
        out[b] = np.average(vals, weights=w) if w.sum() else np.nan
    out = out[np.isfinite(out)]
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return {"mean": float(np.mean(out)), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, p)), "n_boot": int(len(out))}


# ------------------------------------------------------------ permutation test
def permute_years(df, rng):
    """Shuffle the ruler-level years among rulers; fragments inherit their
    ruler's permuted year (within-ruler spread collapses — intended null)."""
    ruler_year = df.groupby("ruler")["year"].mean()
    perm = pd.Series(rng.permutation(ruler_year.values), index=ruler_year.index)
    out = df.copy()
    out["year"] = out.ruler.map(perm).values
    return out


def permutation_test(df, get_feats, sparse, m, perm_draws, n_perm, seed, obs):
    dist = []
    for i in range(n_perm):
        rng = np.random.default_rng(seed + 7_000_000 + i)
        dfp = permute_years(df, rng)
        rp_p = P.eligible_ruler_pairs(dfp)
        plans = build_plans(dfp, m, perm_draws, seed + 9_000_000 + i * 131, rp_p)
        dist.append(macro(score_plans(dfp, get_feats, sparse, plans)))
        if (i + 1) % 25 == 0:
            print(f"    perm {i + 1}/{n_perm}  (last={dist[-1]:.3f})", flush=True)
    dist = np.array(dist)
    return {"n_perm": int(n_perm), "perm_mean": float(dist.mean()),
            "perm_p95": float(np.percentile(dist, 95)),
            "p_value": float((1 + (dist >= obs).sum()) / (n_perm + 1)),
            "note": "layer fixed at F1 best layer (selected on real labels)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(PP.TEXT_COL))
    ap.add_argument("--arms", nargs="+", required=True,
                    help=f"first-listed twin arms pair with their trained arm; "
                         f"'{FLOOR}' is the floor")
    ap.add_argument("--site", default="mean")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--perm-draws", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-perm", nargs="*", default=[],
                    help="arms to exclude from the (expensive) permutation test")
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    rp = P.eligible_ruler_pairs(df)
    rulers = sorted(df.ruler.unique())
    plans = build_plans(df, args.m, args.draws, args.seed, rp)
    print(f"[data] {len(df)} frags, {len(rulers)} rulers, {len(rp)} ruler-pairs, "
          f"{args.draws} shared draws", flush=True)

    os.makedirs(os.path.join(RESULTS, "inference", "rp"), exist_ok=True)
    scores, tables, out = {}, {}, {"variant": args.variant, "m": args.m,
                                   "draws": args.draws, "arms": {}}
    for arm in args.arms:
        fixed = -1
        if arm != FLOOR:
            f1 = os.path.join(RESULTS, "probes",
                              f"{arm}.{args.variant}.{args.site}.json")
            if not os.path.exists(f1):
                sys.exit(f"no F1 result {f1} — run probe_pairs first (the fixed "
                         "layer comes from there).")
            fixed = json.load(open(f1))["best_layer"]
        print(f"[arm] {arm} (layer {fixed})", flush=True)
        get_feats, sparse = arm_features(df, arm, args.variant, args.site, fixed)
        sc = score_plans(df, get_feats, sparse, plans)
        sc.to_csv(os.path.join(RESULTS, "inference", "rp",
                               f"{arm}.{args.variant}.csv.gz"), index=False)
        scores[arm], tables[arm] = sc, rp_table(sc)
        obs = macro(sc)
        rec = {"macro_acc": obs, "fixed_layer": fixed}
        if arm not in args.skip_perm:
            print(f"  permutation ({args.n_perm} x {args.perm_draws} draws)...",
                  flush=True)
            rec["permutation"] = permutation_test(
                df, get_feats, sparse, args.m, args.perm_draws, args.n_perm,
                args.seed, obs)
            print(f"  perm p={rec['permutation']['p_value']:.4f} "
                  f"(obs {obs:.3f} vs null mean "
                  f"{rec['permutation']['perm_mean']:.3f})", flush=True)
        out["arms"][arm] = rec

    # paired contrasts on the shared draws
    contrasts = {}
    for arm in args.arms:
        if arm == FLOOR or arm not in tables:
            continue
        pairs_to = {"vs_floor": FLOOR}
        twin = (arm + "_random" if arm + "_random" in tables
                else "random" if "random" in tables and "qwen" in arm else None)
        if twin:
            pairs_to["vs_twin"] = twin
        for name, other in pairs_to.items():
            if other not in tables:
                continue
            joint = pd.concat([tables[arm].rename("a"),
                               tables[other].rename("b")], axis=1).dropna()
            contrasts[f"{arm}__{name}"] = {
                "other": other, "n_ruler_pairs": int(len(joint)),
                "mean_diff": float((joint.a - joint.b).mean()),
                "dyadic_bootstrap": dyadic_boot(joint.a - joint.b, rulers,
                                                args.n_boot, args.seed)}
            db = contrasts[f"{arm}__{name}"]["dyadic_bootstrap"]
            print(f"[contrast] {arm} {name} ({other}): "
                  f"diff={db['mean']:+.3f} CI{db['ci95']} p={db['p_two_sided']:.4f}",
                  flush=True)
    out["contrasts"] = contrasts

    pth = os.path.join(RESULTS, "inference", f"{args.variant}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
