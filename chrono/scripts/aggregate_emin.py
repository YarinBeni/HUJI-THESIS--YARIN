"""aggregate_emin.py — read the C3 E-MIN score parquets out through the SLA
read-out and put them next to the C2 baseline at the same features.

WHAT. For each seed: take every fold's TEST rows (is_test), centre them by
that fold model's mean TRAIN score (heads trained on different rulers share
no intercept -- the same offset fix run_baseline_gate.cross_fit applies),
concatenate into ONE frozen out-of-fold scoring per (condition, doc), and
run chrono.eval.robustness.battery over {gkf_ruler (pooled), mc_balanced}
plus the ruler-block null on the `orig` condition. Then the C2 baseline
rows (ridge / PLS, same model, layer, site) from results.parquet.

WHY centre and not rank. s_rank is fold-local and would be the alternative,
but ranks concatenated across folds put every fold's median at .5 and
destroy between-fold order; centring by the train mean keeps within-fold
scale and removes only the intercept, which is what differs.

    python chrono/scripts/aggregate_emin.py --run emin_thalesian \
        --baseline-layer 8 --baseline-site mean --lang akk
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from chrono import common                                   # noqa: E402
from chrono.eval.robustness import battery                  # noqa: E402
from chrono.eval.protocol import block_placebo_rho          # noqa: E402


def load_oof(paths: list) -> pd.DataFrame:
    """One centred out-of-fold scoring per (condition, doc) from a seed's
    fold parquets. Raises if any doc is scored twice or never."""
    parts = []
    for p in sorted(paths):
        d = pd.read_parquet(p)
        for cond, g in d.groupby("condition", sort=False):
            tr_mean = g.loc[~g["is_test"], "s"].mean()
            te = g[g["is_test"]].copy()
            te["s"] = te["s"] - tr_mean
            parts.append(te[["doc_id", "condition", "s", "fold"]])
    oof = pd.concat(parts, ignore_index=True)
    dup = oof.duplicated(["condition", "doc_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} (condition, doc) scored in >1 fold")
    return oof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="emin_thalesian")
    ap.add_argument("--scores-dir", default="chrono/reports/scores")
    ap.add_argument("--results", default="chrono/reports/results.parquet")
    ap.add_argument("--corpus", default=os.path.join(common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--splits-dir", default=os.path.join(common.ART, "splits"))
    ap.add_argument("--baseline-model", default="Thalesian/AKK_300m")
    ap.add_argument("--baseline-layer", type=int, default=8)
    ap.add_argument("--baseline-site", default="mean")
    ap.add_argument("--lang", default="akk")
    ap.add_argument("--out", default="chrono/reports/emin_summary.md")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(args.corpus)
    splits = {}
    for name in ("gkf_ruler", "mc_balanced"):
        with open(os.path.join(args.splits_dir, f"{name}.json")) as f:
            splits[name] = json.load(f)

    files = glob.glob(os.path.join(args.scores_dir, f"{args.run}-s*-f*.parquet"))
    seeds = sorted({os.path.basename(f).split("-s")[1].split("-f")[0] for f in files})
    if not seeds:
        raise SystemExit(f"no score files for {args.run} under {args.scores_dir}")

    rows, nulls = [], []
    for sd in seeds:
        oof = load_oof([f for f in files if f"-s{sd}-f" in f])
        n_folds = oof["fold"].nunique()
        b = battery(oof[["doc_id", "condition", "s"]], corpus, splits)
        b["seed"] = int(sd); b["n_folds"] = n_folds
        rows.append(b)
        orig = oof[oof["condition"] == "orig"]
        s = pd.Series(orig["s"].to_numpy(), index=pd.Index(orig["doc_id"], name="doc_id"))
        nulls.append(block_placebo_rho(s, corpus, splits["mc_balanced"], seed=int(sd)))
        if sd == seeds[0]:
            # null of the STATISTIC we report (mean over draws), not of a
            # single draw: B fresh ruler->t permutations, each averaged
            # over the draws exactly as the head's mc rho is
            null_of_mean = np.array([
                np.nanmean(block_placebo_rho(s, corpus, splits["mc_balanced"],
                                             seed=10_000 + b))
                for b in range(200)])
    B = pd.concat(rows, ignore_index=True)
    null = np.concatenate(nulls)

    # per condition x split: mean over seeds of rho_mean, sd across seeds
    agg = (B.groupby(["condition", "split", "readout"], sort=False)["rho_mean"]
             .agg(["mean", "std", "count"]).reset_index())

    # C2 baseline rows at the same features
    R = pd.read_parquet(args.results)
    pre = f"p04_gate::{args.baseline_model}::L{args.baseline_layer}::{args.baseline_site}::"
    base = R[R.run_id.str.startswith(pre) & R.metric.eq("rho_mean")].copy()
    base["probe"] = base.run_id.str.replace(pre, "", regex=False)
    base["lang"] = base["extra"].map(lambda e: json.loads(e).get("lang"))
    base = base[base["lang"] == args.lang]

    lines = [f"# E-MIN summary — `{args.run}`",
             "",
             f"Seeds: {', '.join(seeds)} · folds per seed: "
             f"{sorted(B['n_folds'].unique().tolist())} · read-out: SLA §7 "
             "(gkf pooled over centred OOF scores; mc = mean of per-draw rho)",
             "",
             "## Chrono-Barlow head, by condition (mean ± sd across seeds)",
             "", "| condition | gkf pooled ρ | mc ρ | seeds |", "|---|---|---|---|"]
    for cond in list(dict.fromkeys(agg["condition"])):
        g = agg[agg["condition"] == cond].set_index("split")
        gk = g.loc["gkf_ruler"]; mc = g.loc["mc_balanced"]
        lines.append(f"| `{cond}` | {gk['mean']:+.3f} ± {gk['std']:.3f} | "
                     f"{mc['mean']:+.3f} ± {mc['std']:.3f} | {int(gk['count'])} |")
    lines += ["",
              f"Ruler-block null on `orig` (all seeds × draws): "
              f"{null.mean():+.3f} ± {null.std():.3f}, 95% band "
              f"[{np.quantile(null, .025):+.3f}, {np.quantile(null, .975):+.3f}] "
              "— that is ONE draw; the reported mc ρ is a mean over draws, whose "
              f"block null is {null_of_mean.mean():+.3f} ± {null_of_mean.std():.3f}, "
              f"95% band [{np.quantile(null_of_mean, .025):+.3f}, "
              f"{np.quantile(null_of_mean, .975):+.3f}] (200 ruler permutations, seed 0 scores)",
              "",
              f"## C2 baseline at the same features ({args.baseline_model} "
              f"L{args.baseline_layer} {args.baseline_site}, lang={args.lang})",
              "", "| probe | split | ρ |", "|---|---|---|"]
    for _, r in base.sort_values(["probe", "split"]).iterrows():
        lines.append(f"| {r['probe']} | {r['split']} | {r['value']:+.3f} |")
    lines += ["",
              "Baseline rows are ONE cross-fit (no seeds); the head has 5. "
              "Both use the same folds, the same pooled/mc read-out and the "
              "same corpus, so the columns are directly comparable."]
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
