"""Collect E1 results into one tidy CSV per family.

    python aggregate_pairs.py

Writes results/summary_probes.csv and results/summary_behavioral.csv, one row per
(method, variant, site), sorted so the table reads top-down like the deck tables.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def probes():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "probes", "*.json"))):
        j = json.load(open(p))
        f = j.get("full", {})
        if f.get("skipped"):
            continue
        rows.append({
            "method": j["method"], "variant": j["variant"], "site": j["site"],
            "best_layer": j["best_layer"], "m": j["m"], "draws": j["draws"],
            "macro_acc": f.get("macro_acc_mean"), "macro_sd": f.get("macro_acc_std"),
            "micro_acc": f.get("micro_acc_mean"), "auc": f.get("auc_mean"),
            "auc_sd": f.get("auc_std"),
            "ruler_pairs_tested": f.get("n_ruler_pairs_mean"),
            **{f"acc_d{b.strip('[)').replace(', ', '_')}": v["mean"]
               for b, v in f.get("acc_by_dyear", {}).items()},
        })
    return pd.DataFrame(rows)


def behavioral():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "behavioral", "*.json"))):
        j = json.load(open(p))
        rows.append({k: j.get(k) for k in
                     ("method", "variant", "n_pairs", "n_ruler_pairs",
                      "macro_acc", "micro_acc", "order_consistency",
                      "macro_acc_consistent_only", "yes_rate")})
    return pd.DataFrame(rows)


def main():
    for name, df in (("summary_probes", probes()),
                     ("summary_behavioral", behavioral())):
        if df.empty:
            print(f"[{name}] nothing yet")
            continue
        df = df.sort_values(["variant", "macro_acc"], ascending=[True, False])
        out = os.path.join(RESULTS, f"{name}.csv")
        df.to_csv(out, index=False)
        print(f"[{name}] {len(df)} rows -> {out}")
        cols = [c for c in ("method", "variant", "site", "macro_acc", "macro_sd",
                            "auc", "order_consistency") if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
