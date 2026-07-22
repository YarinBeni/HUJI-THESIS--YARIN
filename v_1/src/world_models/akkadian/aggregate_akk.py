"""WA aggregation: per-target tables across the three modes.

For each target (year, geo) writes an R²/ρ table with, per method×variant:
  r8: holdout R² | balanced-MC ρ | leave-one-ruler-out ρ
  r40: holdout R² | leave-one-ruler-out ρ   (balanced-MC N/A: min ruler count = 1)
The three columns show the collapse: holdout (ruler-ID inflated) -> MC (balanced,
in-distribution) -> LORO (unseen ruler, the real test). Reads both the multi-mode
JSONs and the older holdout-only ones. Canonical site: last (decoders) / text (tfidf).

    python aggregate_akk.py
"""
import glob
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import akk_data as A          # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
METHOD_ORDER = ["llama2_70b", "llama2_13b", "llama2_7b", "gpt_oss_120b",
                "qwen3_32b", "qwen3_8b", "qwen3_1b7",
                "llama2_70b_random", "llama2_13b_random", "llama2_7b_random",
                "random", "tfidf"]


def _get(d, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_rows():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "probes", "*", "*.json")):
        r = json.load(open(path))
        if r.get("site") not in ("last", "text"):   # canonical site only
            continue
        rows.append({
            "method": r["method"], "variant": r["variant"],
            "ruler_set": r["ruler_set"], "target": r["target"],
            "hold_r2": _get(r, "holdout", "best_test_r2",
                            default=r.get("best_test_r2", float("nan"))),
            "mc_rho": _get(r, "mc", "spearman_mean"),
            "loro_rho": _get(r, "loro", "spearman"),
        })
    return pd.DataFrame(rows)


def main():
    df = load_rows()
    if df.empty:
        print("no akkadian probe results yet")
        return
    df["row"] = df.method + " · " + df.variant
    df["__k"] = df.method.apply(lambda m: METHOD_ORDER.index(m)
                                if m in METHOD_ORDER else 99)

    lines = ["# WA results — Akkadian, three modes",
             "",
             "Per method×text-variant. **r8 holdout R²** (within-ruler split, inflated "
             "by ruler identity) → **r8 MC ρ** (balanced, in-distribution, 200 draws) → "
             "**r8 LORO ρ** (leave-one-ruler-out — the real 'place an unseen ruler' "
             "test). r40 MC is N/A (min ruler count = 1). Decoders last-token; encoders "
             "excluded.", ""]

    for target in A.TARGETS:
        sub = df[df.target == target].copy()
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["__k", "row"], columns="ruler_set",
                              values=["hold_r2", "mc_rho", "loro_rho"], aggfunc="first")
        # build a flat, ordered table
        out = pd.DataFrame(index=piv.index)
        out["r8 hold R²"] = piv[("hold_r2", "r8")] if ("hold_r2", "r8") in piv else float("nan")
        out["r8 MC ρ"] = piv[("mc_rho", "r8")] if ("mc_rho", "r8") in piv else float("nan")
        out["r8 LORO ρ"] = piv[("loro_rho", "r8")] if ("loro_rho", "r8") in piv else float("nan")
        out["r40 hold R²"] = piv[("hold_r2", "r40")] if ("hold_r2", "r40") in piv else float("nan")
        out["r40 LORO ρ"] = piv[("loro_rho", "r40")] if ("loro_rho", "r40") in piv else float("nan")
        out = out.reset_index().drop(columns="__k").set_index("row")
        out.round(3).to_csv(os.path.join(RESULTS_DIR, f"summary_{target}_modes.csv"))
        lines += [f"## {target}", "", out.round(3).to_markdown(), ""]

    with open(os.path.join(RESULTS_DIR, "RESULTS_akk.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS_DIR}/RESULTS_akk.md")


if __name__ == "__main__":
    main()
