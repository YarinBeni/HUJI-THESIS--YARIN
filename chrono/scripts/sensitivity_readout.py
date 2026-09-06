"""sensitivity_readout.py — re-read existing scores with documents removed:
(a) duplicate-text documents (keep the first of each byte-identical group,
per language), (b) century-coded documents (t_quality == 'century'), (c)
both. No retraining: the frozen out-of-fold scores are simply restricted,
so this answers "would the number move if we had excluded them".

    python chrono/scripts/sensitivity_readout.py --scores-dir chrono/reports/tier0/scores \
        --art chrono/artifacts_tier0 --runs emin2_cunei400m_t0_akk baseline_ridge_L12mean_akk
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrono.eval.protocol import mc_balanced_rho, pooled_rho   # noqa: E402
from aggregate_emin import load_oof                            # noqa: E402


def keep_sets(corpus):
    dup_drop = set()
    for col in ("text_akk", "text_eng"):
        t = corpus.set_index("doc_id")[col].astype(str)
        t = t[t.str.strip() != ""]
        dup_drop |= set(t.index[t.duplicated(keep="first")])
    century = set(corpus.loc[corpus["t_quality"] == "century", "doc_id"])
    all_ids = set(corpus["doc_id"])
    return {"all": all_ids, "no_dup": all_ids - dup_drop,
            "no_century": all_ids - century, "no_dup_no_century": all_ids - dup_drop - century}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="chrono/reports/tier0/scores")
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--condition", default="orig")
    ap.add_argument("--out", default="chrono/reports/sensitivity_dup_century.md")
    args = ap.parse_args(argv)
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    with open(os.path.join(args.art, "splits", "mc_balanced.json")) as f:
        mc = json.load(f)
    with open(os.path.join(args.art, "splits", "gkf_ruler.json")) as f:
        gkf = json.load(f)
    sets = keep_sets(corpus)
    lines = [f"# Sensitivity: duplicates / century docs (condition `{args.condition}`)", "",
             f"docs: all {len(sets['all'])} · no_dup {len(sets['no_dup'])} · no_century "
             f"{len(sets['no_century'])} · both {len(sets['no_dup_no_century'])}", "",
             "| run | subset | mc ρ (mean over seeds) | gkf pooled ρ |", "|---|---|---|---|"]
    for run in args.runs:
        files = glob.glob(os.path.join(args.scores_dir, f"{run}-s*-f*.parquet"))
        seeds = sorted({re.search(r"-s(\d+)-f", f).group(1) for f in files})
        for name, keep in sets.items():
            mcs, pools = [], []
            for sd in seeds:
                oof = load_oof([f for f in files if f"-s{sd}-f" in f])
                o = oof[(oof["condition"] == args.condition) & oof["doc_id"].isin(keep)]
                s = pd.Series(o["s"].to_numpy(), index=pd.Index(o["doc_id"], name="doc_id"))
                sub = corpus[corpus["doc_id"].isin(keep)]
                mc_k = {"folds": [{"test": [d for d in f_["test"] if d in keep]} for f_ in mc["folds"]]}
                gkf_k = {"folds": [{"train": f_["train"], "test": [d for d in f_["test"] if d in keep]} for f_ in gkf["folds"]]}
                mcs.append(np.nanmean(mc_balanced_rho(s, sub, mc_k)))
                pools.append(pooled_rho(s, sub, gkf_k))
            lines.append(f"| `{run}` | {name} | {np.mean(mcs):+.3f} | {np.mean(pools):+.3f} |")
    txt = "\n".join(lines) + "\n"
    open(args.out, "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
