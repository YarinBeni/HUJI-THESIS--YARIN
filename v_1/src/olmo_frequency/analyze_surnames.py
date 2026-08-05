#!/usr/bin/env python3
"""STEP 3e: does the "never seen" group survive counting surnames?

RESULTS.md §2b claims that 583 people whose full name never appears once are still dated
to within 122 years. That claim is only as strong as the definition of "never appears",
and the exact-full-string definition is weak: "Franz Xaver Feuchtmayer" scores zero while
"Feuchtmayer" may be everywhere.

This re-runs the tail analysis with exposure defined as **max(full name, surname)** — a
deliberately generous upper bound — and reports:

  * how many of the zero group survive as zero under the looser definition
  * the dating error of the survivors, which is the number the claim rests on
  * the same tails as §2b, recomputed, so the two are directly comparable

If almost nobody survives, the honest conclusion is that we never had unseen entities and
§2b's left column has to be withdrawn. If a decent group survives with a similar error,
the claim gets stronger rather than weaker.

    python analyze_surnames.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SUR = os.path.join(RESULTS, "surname_counts.csv")
ARMS = [("olmo2_7b", "trained OLMo"), ("olmo2_7b_random", "random twin")]


def main():
    if not os.path.exists(SUR):
        sys.exit(f"no {SUR} — run count_surnames.py first (needs internet).")
    s = pd.read_csv(SUR)
    s = s[(s["error"].isna()) | (s["error"] == "")]
    s["full_count"] = pd.to_numeric(s["full_count"], errors="coerce")
    s["surname_count"] = pd.to_numeric(s["surname_count"], errors="coerce")
    # the generous reading: whatever the model saw, it saw at least this often
    s["exposure"] = s[["full_count", "surname_count"]].max(axis=1)

    zero_full = s[s.full_count == 0]
    zero_both = zero_full[zero_full.exposure == 0]
    print(f"[surnames] {len(s)} figures with a surname count")
    print(f"  zero on the FULL name          : {len(zero_full)}")
    print(f"  still zero once the surname is counted: {len(zero_both)} "
          f"({100 * len(zero_both) / max(len(zero_full), 1):.1f}% of them)")
    if len(zero_full):
        med = zero_full.surname_count.median()
        print(f"  median surname count among the zero group: {med:.0f}")

    for arm, lab in ARMS:
        j = pd.read_csv(os.path.join(RESULTS, f"joined_{arm}.csv"))
        d = j.merge(s[["name", "full_count", "surname_count", "exposure"]],
                    on="name", how="inner")
        if not len(d):
            print(f"\n{lab}: no overlap with the joined table")
            continue
        p95 = np.percentile(d.exposure, 95)
        zf = d[d.full_count == 0]
        zb = d[d.exposure == 0]
        hi = d[d.exposure >= p95]
        print(f"\n=== {lab}  (n={len(d)})")
        print(f"  zero on full name only  n={len(zf):4d}  median |err| = "
              f"{zf.abs_err.median():6.1f} yr" if len(zf) else "  (no zero-full rows)")
        if len(zb):
            print(f"  zero on BOTH forms      n={len(zb):4d}  median |err| = "
                  f"{zb.abs_err.median():6.1f} yr   <- the claim rests on this")
        else:
            print("  zero on BOTH forms      n=   0   <- §2b's left column does not survive")
        print(f"  top 5% by exposure      n={len(hi):4d}  median |err| = "
              f"{hi.abs_err.median():6.1f} yr")
        if len(zb) and len(hi):
            print(f"  gap: {zb.abs_err.median() - hi.abs_err.median():+.1f} yr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
