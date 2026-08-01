"""WB probing — ridge/PLS probes over the CELL-B entity-level activations.

Why this is not `probe_wm.py`: these datasets have **34 rulers / 25 places**, not the
paper's thousands of entities. A single 20% hold-out is 6-7 test entities, so one
split's R² is mostly noise. We therefore report a **Monte-Carlo over entity-level
splits** (200 draws, 20% of entities held out each time) as the headline number, with
the committed fixed split kept alongside for reference. Splitting by `entity_ix` means
all six templates of a ruler move together — a template of a train entity can never
leak its target into test.

Four pooling sites are scored independently (`ent_last`, `ent_mean`, `last`, `mean`),
and rows are additionally scored **bare-only** so the exact paper-faithful probe (the
entity string alone, entity-last-token) is readable without the carrier sentences.

    python probe_entity.py --method qwen3_8b
    python probe_entity.py --tfidf                    # the n-gram floor, no GPU
    python probe_entity.py --method llama2_70b --cleanup

Writes results/probes_entity/{method}/{entity_type}.{site}.json (committed).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from wm_lib import probing                            # noqa: E402
from wm_lib.registry import MODELS                    # noqa: E402

ENTITY_TYPES = {"assyrian_ruler": ("death_year", False),
                "mesopotamian_place": (("longitude", "latitude"), True)}
SITES = ["ent_last", "ent_mean", "last", "mean"]
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "entity_datasets")
ACTS_DIR = os.environ.get("WM_ACTS_DIR") or os.path.join(os.path.dirname(_HERE), "activations")
RESULTS_DIR = os.path.join(_HERE, "results")
N_DRAWS = 200
TEST_RATIO = 0.2
SEED = 42


def load_df(entity_type):
    return pd.read_csv(os.path.join(DATA_DIR, f"{entity_type}.csv"))


def targets(entity_type, df):
    feat, is_place = ENTITY_TYPES[entity_type]
    if is_place:
        return df[list(feat)].values.astype(float), True
    return df[feat].values.astype(float), False


def _rho(test_scores):
    """Spearman for either target: time probes report `spearman`, place probes report
    per-axis lon/lat. dict.get's default is evaluated eagerly, so this must be a
    branch, not a default expression (it warned on every draw otherwise)."""
    if "spearman" in test_scores:
        return float(test_scores["spearman"])
    vals = [test_scores.get("lon_spearman"), test_scores.get("lat_spearman")]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def mc_entity_scores(X, y, ent_ix, is_place, n_draws=N_DRAWS, k=None):
    """Repeated entity-level splits. Returns dict of mean/sd for r2 and spearman."""
    rng = np.random.RandomState(SEED)
    ents = np.unique(ent_ix)
    n_test = max(1, int(round(len(ents) * TEST_RATIO)))
    r2s, rhos = [], []
    for _ in range(n_draws):
        te_ents = rng.choice(ents, size=n_test, replace=False)
        is_test = np.isin(ent_ix, te_ents)
        if is_test.all() or (~is_test).all():
            continue
        try:
            if k is None:
                sc, _, _ = probing.run_probe(X, y, is_test, is_place)
            else:
                sc, _, _ = probing.run_pls_probe(X, y, is_test, is_place, k=k)
        except Exception:  # noqa: BLE001 — a degenerate draw must not kill the sweep
            continue
        r2s.append(sc["test"]["r2"])
        rhos.append(_rho(sc["test"]))
    if not r2s:
        return None
    return {"mc_r2": float(np.mean(r2s)), "mc_r2_sd": float(np.std(r2s)),
            "mc_rho": float(np.nanmean(rhos)), "mc_rho_sd": float(np.nanstd(rhos)),
            "n_draws": len(r2s)}


#: Same reasoning as akk_modes.PLS_KS — k=5 was an inherited default, not a fitted
#: choice, so sweep it and record the whole curve. `pls5_mc` is kept unchanged for
#: comparability with everything already published off these files.
PLS_KS = (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64)


def score_matrix(X, df, entity_type, tag):
    """All read-outs for one feature matrix: MC ridge, the MC PLS-k sweep (with PLS-5
    retained under its old key), and the fixed holdout."""
    y, is_place = targets(entity_type, df)
    ent_ix = df.entity_ix.values
    is_test = df.is_test.values.astype(bool)
    out = {"n": int(len(df)), "n_entities": int(len(np.unique(ent_ix))), "tag": tag}

    mc = mc_entity_scores(X, y, ent_ix, is_place)
    if mc:
        out["ridge_mc"] = mc
    per_k = {}
    for k in PLS_KS:
        if k >= min(X.shape):
            continue
        s = mc_entity_scores(X, y, ent_ix, is_place, k=k)
        if s:
            per_k[str(k)] = s
    if per_k:
        out["pls_per_k"] = per_k
        bk = max(per_k, key=lambda kk: per_k[kk]["mc_rho"])
        out["pls_best_k"] = int(bk)
        out["pls_best_mc"] = per_k[bk]
        out["pls_k_at_grid_ceiling"] = int(bk) == max(PLS_KS)
    if "5" in per_k:
        out["pls5_mc"] = per_k["5"]
    if is_test.any() and (~is_test).any():
        sc, _, _ = probing.run_probe(X, y, is_test, is_place)
        out["ridge_holdout"] = {"r2": sc["test"]["r2"], "rho": _rho(sc["test"])}
    return out


def probe_one(method, entity_type, site, args):
    act_dir = os.path.join(ACTS_DIR, method, entity_type)
    files = sorted(glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    if not files:
        print(f"[skip] no {site} activations for {method}/{entity_type}")
        return None

    df = load_df(entity_type)
    with open(os.path.join(act_dir, "metadata.json")) as f:
        meta = json.load(f)
    df = df.iloc[:meta["n_rows"]]
    bare = (df.template == "bare").values

    per_layer, best = {}, (None, -np.inf)
    for path in files:
        li = int(re.search(r"layer(\d+)\.npz$", path).group(1))
        X = np.load(path)["acts"][:len(df)]
        X, bad = probing.sanitize(X)
        if bad > 0.01:
            print(f"[warn] layer {li}: {bad:.1%} non-finite, skipping")
            continue
        entry = {"all": score_matrix(X, df, entity_type, "all")}
        if bare.sum() >= 8:
            entry["bare"] = score_matrix(X[bare], df[bare], entity_type, "bare")
        per_layer[li] = entry
        r2 = entry["all"].get("ridge_mc", {}).get("mc_r2", -np.inf)
        if r2 > best[1]:
            best = (li, r2)
        print(f"[{method}/{entity_type}/{site}] layer {li}: "
              f"mc_r2={r2:.3f}", flush=True)

    if not per_layer:
        return None
    out = {"method": method, "entity_type": entity_type, "site": site,
           "protocol": f"entity-level MC ({N_DRAWS} draws, {TEST_RATIO:.0%} of entities)",
           "n_entities": int(df.entity_ix.nunique()),
           "templates": sorted(df.template.unique().tolist()),
           "layers": {str(k): v for k, v in sorted(per_layer.items())},
           "best_layer": best[0], "best_mc_r2": float(best[1])}
    pdir = os.path.join(RESULTS_DIR, "probes_entity", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{entity_type}.{site}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_tfidf(args):
    """Character n-gram floor on the same splits — the control the reading rule needs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    for et in (ENTITY_TYPES if args.entity_type == "all" else [args.entity_type]):
        df = load_df(et)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=1, max_features=20000)
        X = vec.fit_transform(df.entity_string.astype(str)).toarray().astype(np.float32)
        bare = (df.template == "bare").values
        out = {"method": "tfidf", "entity_type": et, "site": "text",
               "protocol": f"entity-level MC ({N_DRAWS} draws, {TEST_RATIO:.0%})",
               "n_entities": int(df.entity_ix.nunique()),
               "layers": {"0": {"all": score_matrix(X, df, et, "all")}}}
        if bare.sum() >= 8:
            out["layers"]["0"]["bare"] = score_matrix(X[bare], df[bare], et, "bare")
        out["best_layer"] = 0
        out["best_mc_r2"] = out["layers"]["0"]["all"].get(
            "ridge_mc", {}).get("mc_r2", float("nan"))
        pdir = os.path.join(RESULTS_DIR, "probes_entity", "tfidf")
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, f"{et}.text.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[tfidf] {et}: mc_r2={out['best_mc_r2']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None, choices=sorted(MODELS))
    ap.add_argument("--tfidf", action="store_true", help="run the n-gram floor instead")
    ap.add_argument("--entity-type", default="all",
                    choices=["all"] + list(ENTITY_TYPES))
    ap.add_argument("--sites", default=None, help="comma list to restrict sites")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    if args.tfidf:
        run_tfidf(args)
        return
    if not args.method:
        ap.error("--method is required unless --tfidf")

    sites = args.sites.split(",") if args.sites else SITES
    ets = list(ENTITY_TYPES) if args.entity_type == "all" else [args.entity_type]
    all_ok = True
    for et in ets:
        for site in sites:
            all_ok = (probe_one(args.method, et, site, args) is not None) and all_ok

    if args.cleanup and all_ok:
        n = 0
        for et in ets:
            for p in glob.glob(os.path.join(ACTS_DIR, args.method, et, "*.npz")):
                os.remove(p)
                n += 1
        print(f"[cleanup] removed {n} npz for {args.method}")
    elif args.cleanup:
        print("[cleanup] SKIPPED: some probes missing/failed; activations kept")


if __name__ == "__main__":
    main()
