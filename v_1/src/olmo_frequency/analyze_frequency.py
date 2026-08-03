#!/usr/bin/env python3
"""STEP 3b: does the probe do better on entities OLMo saw more often?

Joins two things:

  entity_counts.csv          how often each entity appears in OLMo's training data
                             (count_frequencies.py, infini-gram)
  results/projections/...    the probe's per-entity prediction at its best layer
                             (written by probe_wm.py during O2)

and asks one question: is |predicted year - true year| smaller for entities that
appeared more often?

THE CONTROL THAT DECIDES IT. The same join is done for `olmo2_7b_random`, the untrained
twin. Its weights never saw the corpus, so any frequency relationship it shows cannot be
about learning — it would have to come from the entity strings themselves (long famous
names tokenise differently, are longer, etc.). A relationship in the trained arm is only
evidence of learning to the extent it is ABSENT in the twin.

THE CONFOUND THAT WOULD FAKE IT. Old people are both rarer in text and harder to date, so
a naive negative correlation could be pure age. Three guards:

  * within-century: the same correlation computed inside each death-century bin, where
    age is held roughly fixed. Reported per bin and as a sample-weighted mean.
  * held-out only: the probe's training rows are excluded, so this is generalisation
    error, not fit.
  * single-token names: "Adams" collides with the common word, inflating its count. The
    flagged rows are reported both in and out.

Everything is Spearman on log10(count + 1), because counts span orders of magnitude and
we only claim a monotone relationship, not a linear one.

    python analyze_frequency.py
    python analyze_frequency.py --min-bin 40
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
WM = os.path.join(os.path.dirname(HERE), "world_models")
DATA = os.path.join(WM, "data", "entity_datasets")
PROJ = os.path.join(WM, "results", "projections")
RESULTS = os.path.join(HERE, "results")
COUNTS = os.path.join(RESULTS, "entity_counts.csv")
ARMS = ["olmo2_7b", "olmo2_7b_random"]
ET = "historical_figure"


def load_counts():
    if not os.path.exists(COUNTS):
        sys.exit(f"no {COUNTS} — run count_frequencies.py first (needs internet; the "
                 "cluster login node has it, this container does not).")
    df = pd.read_csv(COUNTS)
    df = df[(df["error"].isna() | (df["error"] == "")) & df["count"].notna()]
    return df


def load_errors(arm):
    """Per-entity absolute year error at the arm's best layer, held-out rows only."""
    g = sorted(glob.glob(os.path.join(PROJ, arm, f"{ET}.*.layer*.csv.gz")))
    if not g:
        sys.exit(
            f"no projections for {arm} under {PROJ}/{arm}.\n"
            "probe_wm.py writes them during O2, but O2's commit_push did not include\n"
            "results/projections. On the cluster:\n"
            f"    git add -f v_1/src/world_models/results/projections/{arm} && \\\n"
            '    git commit -m "O2: OLMo best-layer projections" && git push origin main')
    p = g[0]
    pr = pd.read_csv(p)
    pr = pr[pr["is_test"].astype(bool)]              # generalisation, not fit
    names = pd.read_csv(os.path.join(DATA, f"{ET}.csv"))["name"]
    pr["name"] = names.iloc[pr["row"].values].values
    pr["abs_err"] = (pr["pred"] - pr["true"]).abs()
    return pr[["name", "abs_err", "true"]], os.path.basename(p)


def spearman(x, y):
    if len(x) < 8:
        return float("nan"), float("nan"), len(x)
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue), len(x)


def analyse(arm, counts, min_bin, drop_short):
    err, src = load_errors(arm)
    df = counts[counts.entity_type == ET].merge(err, on="name", how="inner")
    if drop_short:
        df = df[df.short_name == 0]
    df = df.assign(logc=np.log10(df["count"].astype(float) + 1.0))

    rho, p, n = spearman(df.logc, df.abs_err)
    out = {"arm": arm, "projection": src, "n": n, "overall_rho": rho, "overall_p": p,
           "median_count": float(df["count"].median()),
           "bins": {}}

    # within-century: age held roughly fixed, so a surviving relationship is about
    # frequency rather than about how far back the entity is
    ws, rs = [], []
    for c, g in df.groupby("century"):
        if len(g) < min_bin:
            continue
        br, bp, bn = spearman(g.logc, g.abs_err)
        out["bins"][int(c)] = {"rho": br, "p": bp, "n": bn}
        if br == br:
            ws.append(bn)
            rs.append(br)
    out["within_century_rho"] = (float(np.average(rs, weights=ws)) if rs
                                 else float("nan"))
    out["within_century_bins"] = len(rs)
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bin", type=int, default=30,
                    help="smallest century bin worth a correlation")
    ap.add_argument("--drop-short-names", action="store_true",
                    help="exclude single-token names, whose counts collide with words")
    args = ap.parse_args()

    counts = load_counts()
    print(f"[counts] {len(counts)} usable rows, index="
          f"{counts['index_used'].iloc[0]}")

    stats_out, frames = {}, {}
    for arm in ARMS:
        s, df = analyse(arm, counts, args.min_bin, args.drop_short_names)
        stats_out[arm] = s
        frames[arm] = df
        print(f"\n=== {arm}  (n={s['n']}, {s['projection']})")
        print(f"  overall         rho={s['overall_rho']:+.3f}  p={s['overall_p']:.2g}")
        print(f"  within-century  rho={s['within_century_rho']:+.3f}  "
              f"({s['within_century_bins']} bins, min n={args.min_bin})")

    tr, ct = stats_out[ARMS[0]], stats_out[ARMS[1]]
    stats_out["gap_overall"] = tr["overall_rho"] - ct["overall_rho"]
    stats_out["gap_within_century"] = (tr["within_century_rho"]
                                       - ct["within_century_rho"])
    stats_out["index_used"] = str(counts["index_used"].iloc[0])
    stats_out["drop_short_names"] = bool(args.drop_short_names)
    stats_out["min_bin"] = args.min_bin

    # The count is only from the OLMo corpus, so it is not a training-data count for the
    # twin at all — the twin arm asks "does the entity STRING predict error", which is
    # exactly the null this experiment needs.
    print(f"\ntrained minus twin:  overall {stats_out['gap_overall']:+.3f}   "
          f"within-century {stats_out['gap_within_century']:+.3f}")
    print("  (negative rho = more frequent entities are dated better)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "frequency_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    for arm, df in frames.items():
        df.to_csv(os.path.join(RESULTS, f"joined_{arm}.csv"), index=False)
    print(f"\n[write] {RESULTS}/frequency_stats.json + joined_*.csv")

    # the assyrian rulers are not in the cell-A projections; report them as the
    # descriptive sidebar they are — evidence for the "obscure" label itself
    ak = counts[counts.entity_type == "assyrian_ruler"]
    if len(ak):
        hf = counts[counts.entity_type == ET]["count"].astype(float)
        print(f"\n[sidebar] Assyrian rulers: median count {ak['count'].median():.0f} "
              f"vs historical figures {hf.median():.0f} "
              f"({ak['count'].median() / max(hf.median(), 1):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
