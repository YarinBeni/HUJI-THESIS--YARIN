"""E3 — frozen name-direction transfer + the LEACE mediation test.

THE QUESTION. Cell A proved a year direction exists for ENTITY NAMES (ridge on
name activations, rho ~.88). Is the document-side time axis the SAME axis, just
weaker — or a different axis? And when the frozen direction does order fragments,
does it do so THROUGH ruler identity, or independently of it?

ZERO document-side fitting. The cell-A ridge coefficient vector (saved by
probe_wm.py at its best layer, raw-activation coordinates) is applied as-is to
fragment activations: s = coef . x. Nothing is trained on fragments, so there is
nothing to leak — every fragment is evaluation data. Read-outs:

  * Spearman(s, year) across fragments;
  * the E1 pairwise evaluation with s as the scorer (macro accuracy over
    ruler-pairs, same draws protocol) — directly comparable to the E1 table,
    where every trained probe HAD to be fitted on fragments;
  * both again after LEACE erasure of one-hot ruler identity from the fragment
    activations (concept-erasure, Belrose et al. 2023). Collapse under erasure
    = the transfer was an identity lookup; survival = a ruler-independent time
    component. (By the ICC=1 degeneracy this is a MEDIATION test, not a "does
    year survive" test — see DECIDED_EXPERIMENTS.md F2.)

Layer handling: the frozen direction lives at the cell-A best layer L*. Primary
read-out applies it at the SAME residual-stream depth L* of the fragment run;
a full sweep over fragment layers is reported as exploratory.

Also computed when E1 directions exist: cosine between this frozen cell-A
direction and E1's pairwise direction (trained on relative order only, no
absolute years). Both are moved into the same standardized coordinates
(w_scaled = coef * sd_features over fragments) before the cosine; cross-layer
cosines are labeled as such.

    python e3_transfer.py --method olmo2_7b --variant akk_maximal
    python e3_transfer.py --method olmo2_7b --variant akk_maximal --skip-leace

Writes results/{method}.{variant}.{site}.json. Needs the akkadian npz store and
world_models/results/directions/{method}/ (both cluster-local).
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
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402

_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
DIRS_ROOT = os.path.join(_WM, "results", "directions")
RESULTS = os.path.join(_HERE, "results")

# E3 transfers the cell-A axis (famous figures). E3b transfers the CELL-B axis
# (our 34 rulers, probed as names) — the axis the reviewer argument says is the
# one actually connected to these documents. Cell B has no saved direction
# (probe_entity.py reports Monte-Carlo scores, never a canonical fit), so it is
# fitted here once on the canonical train split and cached as an npz.
# ysign: the polarity convention. Cell-A death years are CE-signed (larger =
# LATER, -935..2021); the ruler CSV stores BC-positive years (Ashurbanipal =
# 631, larger = EARLIER) — the same convention as the fragments' year column.
# The E1 polarity rule in pairwise_eval ("a earlier <=> s_a < s_b") was built
# for later-increasing scorers, so the ruler fit target is NEGATED once here
# and every downstream read-out (spearman, pairwise, LEACE, cosines) stays in
# the same frame as the cell-A rows.
ENTITY_CFG = {
    "historical_figure": dict(
        csv=os.path.join(_WM, "data", "entity_datasets",
                         "historical_figure.csv"),
        acts=os.path.join(_WM, "activations", "{method}",
                          "historical_figure"),
        site="last", ysign=+1.0),
    # NB: akkadian/extract_entity.py writes to world_models/activations/
    # (ACTS_DIR = dirname(akkadian)/activations), NOT akkadian/activations —
    # the same root the cell-A prompts use. Run 25046 extracted fine and
    # then failed the lookup because this pointed one directory too deep.
    "assyrian_ruler": dict(
        csv=os.path.join(_WM, "data", "entity_datasets",
                         "assyrian_ruler.csv"),
        acts=os.path.join(_WM, "activations", "{method}", "assyrian_ruler"),
        site="ent_last", ysign=-1.0),
}
RIDGE_ALPHAS = np.logspace(-1, 6, 15)


def _cv_rho(X, y, groups):
    """5-fold grouped OOF Spearman — layer selection for the cell-B fit."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        pred[te] = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr], y[tr]).predict(X[te])
    return float(stats.spearmanr(pred, y).correlation)


def fit_entity_direction(method, entity, dirs_root):
    """Fit + cache the ruler-axis npz in probe_wm's format so the lens /
    spectroscopy scripts pick it up through their existing globs."""
    cfg = ENTITY_CFG[entity]
    ent = pd.read_csv(cfg["csv"])
    acts_dir = cfg["acts"].format(method=method)
    files = sorted(glob.glob(os.path.join(acts_dir,
                                          f"{cfg['site']}.layer*.npz")))
    if not files:
        sys.exit(f"no {entity} activations under {acts_dir} — run "
                 f"akkadian/extract_entity.py --method {method}")
    y = cfg["ysign"] * ent.death_year.values.astype(float)
    if cfg["ysign"] < 0:
        print("[fitB] BC-positive ruler years negated -> later-increasing "
              "target (cell-A polarity frame)", flush=True)
    valid = np.isfinite(y)
    groups = (ent.entity_ix.values if "entity_ix" in ent
              else np.arange(len(ent)))
    bl = None
    for g in glob.glob(os.path.join(_WM, "akkadian", "results",
                                    "probes_entity", method,
                                    f"{entity}.{cfg['site']}*.json")):
        bl = json.load(open(g)).get("best_layer", bl)
    if bl is None:                 # no committed probe result: pick honestly
        best = (None, -np.inf)
        for p in files:
            L = int(re.search(r"layer(\d+)\.npz$", p).group(1))
            X = np.load(p)["acts"].astype(np.float32)
            r = _cv_rho(X[valid], y[valid], groups[valid])
            print(f"  [fitB] layer {L}: grouped-CV rho {r:+.3f}", flush=True)
            if r > best[1]:
                best = (L, r)
        bl = best[0]
    p = os.path.join(acts_dir, f"{cfg['site']}.layer{bl}.npz")
    X = np.load(p)["acts"].astype(np.float32)
    from sklearn.linear_model import RidgeCV
    tr = valid & ~ent.is_test.astype(bool).values
    probe = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr], y[tr])
    dd = os.path.join(dirs_root, method)
    os.makedirs(dd, exist_ok=True)
    np.savez_compressed(
        os.path.join(dd, f"{entity}.{cfg['site']}.layer{bl}.npz"),
        coef=np.asarray(probe.coef_, np.float32),
        intercept=np.atleast_1d(probe.intercept_).astype(np.float32))
    print(f"[fitB] cached {entity} direction at layer {bl} "
          f"(train n={int(tr.sum())})", flush=True)


def find_entity_direction(method, dirs_root, entity):
    """The saved best-layer ridge direction for the entity set. probe_wm saves
    one file per entity_type x site: {entity}.{site}.layer{L}.npz."""
    pat = os.path.join(dirs_root, method, f"{entity}.*.layer*.npz")
    g = sorted(glob.glob(pat))
    if not g and entity == "assyrian_ruler":
        fit_entity_direction(method, entity, dirs_root)
        g = sorted(glob.glob(pat))
    if not g:
        sys.exit(f"no {entity} direction for {method} under "
                 f"{dirs_root}/{method}.\nprobe_wm.py writes them; on the "
                 f"cluster:  python v_1/src/world_models/probe_wm.py "
                 f"--method {method} --entity-type {entity}")
    p = g[0]                       # one best-layer file per site; site order: any
    m = re.search(rf"{entity}\.(\w+)\.layer(\d+)\.npz$", p)
    z = np.load(p)
    coef = np.asarray(z["coef"], np.float32).ravel()
    return coef, m.group(1), int(m.group(2)), os.path.basename(p)


def pairwise_eval(df, s, m, draws, seed):
    """E1's evaluation with a FIXED scorer: macro accuracy over ruler-pairs.
    No training -> the fold machinery is unnecessary; every drawn pair scores."""
    rp = P.eligible_ruler_pairs(df)
    accs = []
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        pairs = P.draw_pairs(df, m, rng, rp)
        pred = (s[pairs.pos_a.values] < s[pairs.pos_b.values])
        # scorer polarity: s is monotone in predicted year, so "a earlier" means
        # s_a < s_b. Sign errors show up as acc < .5, which is itself reported.
        correct = (pred.astype(int) == pairs.label.values).astype(float)
        t = pd.DataFrame({"c": correct,
                          "ra": np.minimum(pairs.ruler_a, pairs.ruler_b),
                          "rb": np.maximum(pairs.ruler_a, pairs.ruler_b)})
        accs.append(t.groupby(["ra", "rb"])["c"].mean().mean())
    return float(np.mean(accs)), float(np.std(accs))


def leace_erase(X, rulers):
    try:
        import torch
        from concept_erasure import LeaceEraser
    except ImportError:
        sys.exit("pip install concept-erasure (and torch) for the mediation test,"
                 " or pass --skip-leace")
    codes = pd.Categorical(rulers).codes
    Z = np.eye(codes.max() + 1, dtype=np.float64)[codes]
    # float64: with d=4096 >> n=1187 the whitening is ill-conditioned in fp32 —
    # the v1 run's 82% norm change on qwen3_8b was likely partly numeric
    Xt = torch.from_numpy(np.ascontiguousarray(X.astype(np.float64)))
    eraser = LeaceEraser.fit(Xt, torch.from_numpy(Z))
    return eraser(Xt).float().numpy()


def ruler_probe_acc(X, rulers, seed=0):
    """Held-out linear ruler-classification accuracy — the surgical check.
    Before erasure it should be high; after a working erasure, near chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    y = pd.Categorical(rulers).codes
    # keep classes with enough members for 3 folds
    ok = pd.Series(y).groupby(y).transform("size").values >= 3
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, C=0.1))
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X[ok], y[ok], cv=cv).mean())


def entity_positive_control(method, entity, coef, LA, site_dir):
    """Score the entity activations with the very same frozen direction
    through the very same code path. If this does not reproduce the entity-set
    held-out rho, any 'transfer fails' claim is a pipeline bug, not a
    finding."""
    cfg = ENTITY_CFG[entity]
    acts = os.path.join(cfg["acts"].format(method=method),
                        f"{site_dir}.layer{LA}.npz")
    if not (os.path.exists(cfg["csv"]) and os.path.exists(acts)):
        return {"skipped": f"missing {acts}"}
    ent = pd.read_csv(cfg["csv"])
    X = np.load(acts)["acts"].astype(np.float32)
    if len(X) != len(ent):
        return {"skipped": "row mismatch"}
    y = cfg["ysign"] * ent["death_year"].values.astype(float)
    te = ent["is_test"].astype(bool).values & np.isfinite(y)
    s = X[te] @ coef
    return {"spearman_heldout": spearman(s, y[te]), "n": int(te.sum())}


def stability(method, entity, LA, site_dir, coef_full, frag_X, year, df,
              m, draws, seed, K):
    """THE 'WHICH w DID YOU TAKE' ANSWER, as data. The canonical direction is
    a single fit on the fixed train split; the Monte-Carlo machinery was
    evaluation-only. Here we refit the direction on K resampled 80% train
    sets (grouped by entity) and push EVERY refit through the frozen
    transfer, reporting the spread. If the conclusion depended on which fit
    you took, these quantiles would say so."""
    cfg = ENTITY_CFG[entity]
    p = os.path.join(cfg["acts"].format(method=method),
                     f"{site_dir}.layer{LA}.npz")
    if not os.path.exists(p):
        return {"skipped": p}
    from sklearn.linear_model import RidgeCV
    ent = pd.read_csv(cfg["csv"])
    Xe = np.load(p)["acts"].astype(np.float32)
    y = cfg["ysign"] * ent.death_year.values.astype(float)
    valid = np.isfinite(y)
    groups = (ent.entity_ix.values if "entity_ix" in ent
              else np.arange(len(ent)))
    ug = np.unique(groups[valid])
    rng = np.random.default_rng(seed)
    cosims, rhos, macs = [], [], []
    for k in range(K):
        keep = rng.choice(ug, size=int(0.8 * len(ug)), replace=False)
        tr = valid & np.isin(groups, keep)
        w = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xe[tr], y[tr]).coef_.ravel()
        cosims.append(float(w @ coef_full /
                            (np.linalg.norm(w) * np.linalg.norm(coef_full)
                             + 1e-9)))
        s = frag_X @ w.astype(np.float32)
        rhos.append(spearman(s, year))
        macs.append(pairwise_eval(df, s, m, min(draws, 10), seed + 977 + k)[0])
    q = lambda a: {"q05": float(np.quantile(a, .05)),               # noqa: E731
                   "median": float(np.median(a)),
                   "q95": float(np.quantile(a, .95))}
    return {"K": K, "cos_vs_canonical": q(cosims),
            "frozen_spearman": q(rhos), "pairwise_macro": q(macs)}


def spearman(a, b):
    return float(stats.spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal", choices=list(PP.TEXT_COL))
    ap.add_argument("--site", default="mean",
                    help="fragment pooling site the direction is applied to")
    ap.add_argument("--dirs-root", default=DIRS_ROOT)
    ap.add_argument("--acts-root", default=None,
                    help="override the akkadian activation store (testing)")
    ap.add_argument("--m", type=int, default=P.M_DEFAULT)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--skip-leace", action="store_true")
    ap.add_argument("--seed", type=int, default=P.SEED)
    ap.add_argument("--entity-set", default="historical_figure",
                    choices=list(ENTITY_CFG),
                    help="whose axis transfers: cell A (default) or the "
                    "cell-B ruler axis (E3b)")
    ap.add_argument("--stability", type=int, default=0, metavar="K",
                    help="refit the direction on K resampled train sets and "
                    "report the spread of every read-out")
    args = ap.parse_args()

    if args.acts_root:
        PP.ACTS = args.acts_root
    ENTITY = args.entity_set
    coef, srcA_site, LA, src = find_entity_direction(
        args.method, args.dirs_root, ENTITY)
    df = P.load_eligible()
    layers = PP.load_act_layers(args.method, args.variant, args.site, stride=1)
    year = df.year.values.astype(float)
    print(f"[dir] {src}: cell-A best layer {LA} (site {srcA_site}), "
          f"d={len(coef)} | fragment layers on disk: {len(layers)}", flush=True)

    def readout(X, tag):
        # POLARITY. `year` is BC-positive (larger = earlier) while the entity
        # targets are CE-signed/negated to lateness, so a direction that
        # orders documents CORRECTLY scores rho < 0 and macro < .5 in the raw
        # frame. Both raw keys are kept (every earlier result file uses them)
        # and the lateness-frame values are stored alongside; always compare
        # magnitudes against the untrained twin, never against .5.
        s = X @ coef
        rho = spearman(s, year)
        mac, sd = pairwise_eval(df, s, args.m, args.draws, args.seed)
        print(f"  [{tag}] spearman={rho:+.3f}  pairwise macro={mac:.3f}±{sd:.3f}",
              flush=True)
        return {"spearman": rho, "pairwise_macro": mac, "pairwise_sd": sd,
                "spearman_lateness": -rho, "pairwise_macro_lateness": 1 - mac}

    out = {"method": args.method, "variant": args.variant, "site": args.site,
           "cellA_direction": src, "cellA_layer": LA, "entity_set": ENTITY,
           "m": args.m, "draws": args.draws, "n_fragments": int(len(df))}

    # positive control FIRST: the same direction on its home turf must work,
    # or every downstream null is uninterpretable
    out["positive_control_cellA"] = entity_positive_control(
        args.method, ENTITY, coef, LA, srcA_site)
    print(f"  [control] {ENTITY} held-out: {out['positive_control_cellA']}",
          flush=True)

    # primary: the same residual depth the direction was fitted at
    if LA not in layers:
        near = min(layers, key=lambda L: abs(L - LA))
        print(f"[warn] fragment layer {LA} not on disk; using nearest {near}",
              flush=True)
        LA = near
        out["cellA_layer_used"] = LA
    X = layers[LA]
    out["frozen"] = readout(X, f"frozen L{LA}")

    if args.stability:
        out["stability"] = stability(
            args.method, ENTITY, LA, srcA_site, coef, X, year, df,
            args.m, args.draws, args.seed, args.stability)
        print(f"  [stability] {out['stability']}", flush=True)

    if not args.skip_leace:
        Xe = leace_erase(X, df.ruler.values)
        out["frozen_after_leace_ruler"] = readout(Xe, f"LEACE(ruler) L{LA}")
        # surgical controls: (a) the vectors should barely move; (b) a linear
        # ruler probe must fall to ~chance AFTER while being high BEFORE —
        # otherwise the erasure did not actually erase
        delta = float(np.linalg.norm(Xe - X) / np.linalg.norm(X))
        out["leace_relative_change"] = delta
        out["ruler_probe_acc_before"] = ruler_probe_acc(X, df.ruler.values)
        out["ruler_probe_acc_after"] = ruler_probe_acc(Xe, df.ruler.values)
        print(f"  [leace] rel change {delta:.4f} | ruler-probe acc "
              f"{out['ruler_probe_acc_before']:.3f} -> "
              f"{out['ruler_probe_acc_after']:.3f}", flush=True)

    # exploratory: the frozen direction against every fragment layer
    out["layer_sweep"] = {int(L): spearman(Xl @ coef, year)
                          for L, Xl in layers.items()}

    # cosine vs the E1 pairwise direction (trained on order only), in the
    # standardized coordinates the pairwise probe actually lived in
    cos = {}
    for p in sorted(glob.glob(os.path.join(
            _PAIRS, "results", "directions",
            f"{args.method}.{args.variant}.{args.site}*.npz"))):
        mL = re.search(r"layer(-?\d+)\.npz$", p)
        Lp = int(mL.group(1))
        if Lp not in layers:
            continue
        w_pair = np.load(p)["w"].astype(np.float32)
        sd_feat = layers[Lp].std(axis=0) + 1e-8
        wA = coef * sd_feat            # cell-A direction in scaled coords
        c = float(wA @ w_pair / (np.linalg.norm(wA) * np.linalg.norm(w_pair)))
        cos[os.path.basename(p)] = {
            "cosine": c, "pairwise_layer": Lp, "cellA_layer": LA,
            "cross_layer": Lp != LA}
        print(f"  [cosine] vs {os.path.basename(p)}: {c:+.3f}"
              f"{'  (cross-layer)' if Lp != LA else ''}", flush=True)
    out["cosine_vs_pairwise_direction"] = cos

    os.makedirs(RESULTS, exist_ok=True)
    suffix = "" if ENTITY == "historical_figure" else f".{ENTITY}"
    pth = os.path.join(RESULTS,
                       f"{args.method}.{args.variant}.{args.site}"
                       f"{suffix}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
