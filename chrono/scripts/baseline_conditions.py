"""baseline_conditions.py — the frozen probes read out on the SAME view
conditions as the Chrono-Barlow head, so the plan's robustness claim
("degrades <= half as much as PLS under name-masking / formula-removal")
becomes computable.

WHAT. Same features (EmbStore, model/layer/site from the E-MIN config),
same gkf_ruler folds. Per fold: fit ridge / PLS(k) on the TRAIN docs'
`orig` views (every language and view seed present, exactly the rows the
head trains on for that chain), score EVERY view of EVERY doc, average
per (doc, augs chain) — mirroring train_cjb._condition_scores, which
also averages across languages — and write one parquet per fold in the
head's score schema (run_id, doc_id, condition, s, fit, fold, is_test,
s_rank), so aggregate_emin.py reads it with --run baseline_<probe>.

CAVEAT recorded here because it matters: _condition_scores collapses
languages, so both the head's and these baselines' per-condition scores
mix the Akkadian and the English-gloss views of a document. C2's gate
rows were Akkadian-only, hence not the like-for-like comparison for the
head; these are. A per-language read-out needs `lang` in the score
schema (next revision).

    python chrono/scripts/baseline_conditions.py \
        --config chrono/configs/emin_thalesian.yaml --probes ridge pls
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline_gate import fit_predict                       # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="chrono/configs/emin_thalesian.yaml")
    ap.add_argument("--probes", nargs="+", default=["ridge", "pls"])
    ap.add_argument("--pls-components", type=int, default=2)
    ap.add_argument("--views", default=os.path.join(common.ART, "views.parquet"))
    ap.add_argument("--corpus", default=os.path.join(common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--splits-dir", default=os.path.join(common.ART, "splits"))
    ap.add_argument("--store-root", default=os.path.join(common.ART, "emb_store"))
    ap.add_argument("--out-dir", default="chrono/reports/scores")
    ap.add_argument("--train-views", choices=["orig", "all"], default="orig",
                    help="fit on the train docs' orig views only (a probe) or "
                         "on ALL their views (augmentation as plain data "
                         "augmentation, no invariance loss -- isolates what "
                         "the Barlow objective adds over seeing the views)")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="restrict views to these languages (default: all, "
                         "= what the head sees)")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    feats = cfg["features"]
    views = pd.read_parquet(args.views)
    if args.langs:
        views = views[views["lang"].isin(args.langs)].reset_index(drop=True)
    corpus = pd.read_parquet(args.corpus)
    t = corpus.set_index("doc_id")["t"].astype(float)
    with open(os.path.join(args.splits_dir, "gkf_ruler.json")) as f:
        gkf = json.load(f)
    store = EmbStore(args.store_root)
    X = store.get(feats["model"], feats["layer"], feats["site"],
                  views["view_id"].tolist())
    doc = views["doc_id"].to_numpy()
    chain = views["augs"].fillna("").to_numpy()
    text = views["text"].astype(str).to_numpy()
    is_orig = chain == ""
    all_ids = corpus["doc_id"].tolist()
    tag = f"L{feats['layer']}{feats['site']}" + (
        "" if not args.langs else "_" + "+".join(args.langs)) + (
        "" if args.train_views == "orig" else "_allviews")
    os.makedirs(args.out_dir, exist_ok=True)

    for probe in args.probes:
        run = f"baseline_{probe}_{tag}"
        for k, fold in enumerate(gkf["folds"]):
            tr, te = set(fold["train"]), set(fold["test"])
            fit_mask = is_orig if args.train_views == "orig" else np.ones_like(is_orig)
            tr_rows = np.flatnonzero(fit_mask & np.isin(doc, list(tr)))
            # BUG FIX (runner job 007): the orig chain exists once per view
            # seed with byte-identical text, so every training doc appeared
            # twice per language. RidgeCV's efficient LOO then sees each
            # row's twin, judges tiny alphas as excellent, and returns a
            # memorising probe -- akk-only orig read .094 where C2's single-
            # row fit read .287, and MASKING the names "helped" (.208).
            # One row per distinct training text.
            _, keep = np.unique(text[tr_rows], return_index=True)
            tr_rows = tr_rows[np.sort(keep)]
            ytr = t.loc[doc[tr_rows]].to_numpy()
            s_all, _ = fit_predict(probe, X[tr_rows], ytr, X, args.pls_components)
            frames = []
            for ch in pd.unique(chain):
                cond = "orig" if ch == "" else str(ch)
                m = chain == ch
                per_doc = (pd.Series(s_all[m], index=doc[m])
                           .groupby(level=0).mean().reindex(all_ids))
                frames.append(pd.DataFrame({
                    "run_id": f"{run}-s0-f{k}", "doc_id": all_ids,
                    "condition": cond, "s": per_doc.to_numpy(),
                    "fit": "oof", "fold": k,
                    "is_test": [d in te for d in all_ids]}))
            sc = pd.concat(frames, ignore_index=True)
            sc["s_rank"] = np.nan
            path = os.path.join(args.out_dir, f"{run}-s0-f{k}.parquet")
            sc.to_parquet(path, index=False)
            print(f"[baseline] {run} fold {k}: fit on {len(tr_rows)} distinct {args.train_views} views "
                  f"of {len(tr)} docs -> {path}", flush=True)


if __name__ == "__main__":
    main()
