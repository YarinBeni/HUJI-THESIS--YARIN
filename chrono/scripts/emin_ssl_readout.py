"""Every representation read out on the THESIS protocol, not on periods.

Predicting a period is easy and, in this corpus, nearly the same question as
predicting the source. The experiment that matters is the one the M.Sc. work
already ran: the 1,193 dated royal inscriptions, ruler-grouped folds, ridge on
frozen features, Spearman under SLA §7 (mc_balanced draws + the gkf pooled
read-out). This script runs exactly that, with each cell of the embedding
store as the features -- frozen encoders, SSL adapters, from-scratch and
hybrid encoders side by side, all comparable with EMIN2_RESULT.md.

The SSL runs trained on `split == "train"` only, so none of them ever saw a
dated inscription; the labels enter only through the cross-fitted ridge,
inside the same folds the thesis uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.eval.robustness import battery                      # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline_gate import fit_predict                       # noqa: E402


def filter_splits(splits: dict, keep: set) -> dict:
    """Restrict every fold to the docs we actually have features for."""
    out = {}
    for name, sp in splits.items():
        folds = []
        for f in sp["folds"]:
            tr = [d for d in f["train"] if d in keep]
            te = [d for d in f["test"] if d in keep]
            if len(tr) >= 20 and len(te) >= 5:
                folds.append({**f, "train": tr, "test": te})
        out[name] = {**sp, "folds": folds}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--store-root", default="chrono/artifacts_ssl/emb_store")
    ap.add_argument("--uid-prefix", default="ssl::orcc::",
                    help="how a tier0 doc_id is addressed in the SSL store")
    ap.add_argument("--out", default="chrono/reports/EMIN_SSL_RESULT.md")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    splits = {n: json.load(open(os.path.join(args.art, "splits", f"{n}.json")))
              for n in ("gkf_ruler", "mc_balanced")}
    store = EmbStore(args.store_root)
    man = store.manifest()
    if man.empty:
        raise SystemExit("empty embedding store")

    rows = []
    for model, layer, site in sorted({(r.model, int(r.layer), r.site) for r in man.itertuples()}):
        ids = [args.uid_prefix + d for d in corpus.doc_id]
        try:
            have = np.asarray(store.has(model, layer, site, ids))
        except Exception as exc:                                # noqa: BLE001
            print(f"[skip] {model} L{layer} {site}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if have.sum() < 500:
            continue
        cdf = corpus[have].reset_index(drop=True)
        X = store.get(model, layer, site, [args.uid_prefix + d for d in cdf.doc_id]).astype(np.float32)
        t = cdf.set_index("doc_id")["t"].astype(float)
        keep = set(cdf.doc_id)
        sp = filter_splits(splits, keep)
        if not sp["gkf_ruler"]["folds"]:
            continue

        # cross-fitted OOF scores, centred per fold like load_oof does
        pos = {d: i for i, d in enumerate(cdf.doc_id)}
        parts = []
        for k, fold in enumerate(sp["gkf_ruler"]["folds"]):
            itr = [pos[d] for d in fold["train"]]
            ite = [pos[d] for d in fold["test"]]
            s_te, tr_mean = fit_predict("ridge", X[itr], t.loc[fold["train"]].to_numpy(), X[ite], 2)
            parts.append(pd.DataFrame({"doc_id": fold["test"], "condition": "orig",
                                       "s": np.asarray(s_te).ravel() - tr_mean}))
        oof = pd.concat(parts, ignore_index=True).drop_duplicates(["condition", "doc_id"])
        b = battery(oof, cdf, sp)
        b = b[b.condition == "orig"]
        get = lambda s: (b.loc[b.split == s, "rho_mean"].iloc[0] if (b.split == s).any() else np.nan)
        rows.append(dict(model=f"{model}::L{layer}::{site}", n=int(have.sum()),
                         mc=get("mc_balanced"), gkf=get("gkf_ruler")))
        print(f"{model} L{layer} {site}: n={int(have.sum())} mc={rows[-1]['mc']:.3f} "
              f"gkf={rows[-1]['gkf']:.3f}", flush=True)

    if not rows:
        raise SystemExit("no cell had embeddings for the dated inscriptions")
    R = pd.DataFrame(rows).sort_values("gkf", ascending=False)
    lines = ["# The dated inscriptions, read out on the thesis protocol", "",
             "Ridge on frozen features of the 1,193 dated royal inscriptions, ruler-grouped folds, "
             "SLA §7: `gkf` is the POOLED Spearman over the held-out docs (per-fold rho is undefined "
             "when 39 of 40 rulers carry one year), `mc` the mean over the frozen balanced draws. "
             "Same protocol and same folds as `EMIN2_RESULT.md`, so the rows are directly comparable; "
             "the only thing that changes between rows is which representation the ridge sees. "
             "No SSL run ever trained on these documents.", "",
             "| representation | n docs | mc rho | gkf rho (pooled) |", "|---|---|---|---|"]
    for _, r in R.iterrows():
        lines.append(f"| `{r.model}` | {r.n:,} | {r.mc:.3f} | {r.gkf:.3f} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(R)} cells)")


if __name__ == "__main__":
    main()
