"""WA aggregation: collect probe JSONs into per-(target x ruler_set) tables.

For each target (year, geo) x ruler_set (r8, r40) writes an R2 and a Spearman CSV
with rows = method x text-variant, using the canonical `last` site for decoders and
`text` for tfidf. Also writes RESULTS_akk.md.

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


def load_all():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "probes", "*", "*.json")):
        with open(path) as f:
            r = json.load(f)
        rows.append({k: r[k] for k in ("method", "variant", "ruler_set", "target",
                                       "site", "best_test_r2", "best_test_spearman")})
    return pd.DataFrame(rows)


def main():
    df = load_all()
    if df.empty:
        print("no akkadian probe results yet")
        return
    # canonical site: last for models, text for tfidf
    df = df[(df.site == "last") | (df.method == "tfidf")].copy()
    df["row"] = df.method + " · " + df.variant
    lines = ["# WA results — Akkadian space & time, G&T protocol",
             "",
             "Best-layer held-out **test** scores. Entity = whole fragment (last-token "
             "for decoders); target = year or find-spot (lon,lat); held-out-by-ruler "
             "split. Encoders excluded (no causal last token).", ""]
    order = [m for m in METHOD_ORDER]

    def pivot(sub, val):
        t = sub.pivot_table(index="row", columns="ruler_set", values=val,
                            aggfunc="first")
        t = t.reindex(columns=[c for c in ["r8", "r40"] if c in t.columns])
        # sort by method order then variant
        t["__k"] = [order.index(r.split(" · ")[0]) if r.split(" · ")[0] in order
                    else 99 for r in t.index]
        return t.sort_values("__k").drop(columns="__k")

    for target in A.TARGETS:
        for metric, val in [("R²", "best_test_r2"), ("Spearman ρ", "best_test_spearman")]:
            sub = df[df.target == target]
            if sub.empty:
                continue
            t = pivot(sub, val)
            t.to_csv(os.path.join(RESULTS_DIR, f"summary_{target}_{val}.csv"))
            lines += [f"## {target} — {metric}", "", t.round(3).to_markdown(), ""]
    with open(os.path.join(RESULTS_DIR, "RESULTS_akk.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS_DIR}/RESULTS_akk.md")


if __name__ == "__main__":
    main()
