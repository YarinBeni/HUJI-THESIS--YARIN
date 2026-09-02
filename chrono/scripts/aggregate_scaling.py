"""aggregate_scaling.py — S4: one table for the whole representation sweep.

Reads results.parquet rows written by probe_representations.py
(run_id 's1_probe::<model>::L<layer>::<site>') and train_ssl_e2e.py
(run_id 'ssl_e2e::<run>', metric quick_period_probe with step in extra), and
lays out: frozen encoders vs SSL adapters vs from-scratch sizes on the period
probe (linear / MLP), within-source and held-out-source period probes, the
SOURCE probe (bias), silhouette p-value and k-NN purity; plus the quick-probe
learning curve per from-scratch run and a params-vs-quality table.

    python chrono/scripts/aggregate_scaling.py --out chrono/reports/SCALING_RESULT.md
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def kind_of(model: str) -> str:
    if model.startswith("ssl_e2e::"): return "from-scratch"
    if model.startswith("ssl::"): return "adapter (SSL on frozen)"
    return "frozen encoder"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="chrono/reports/results.parquet")
    ap.add_argument("--out", default="chrono/reports/SCALING_RESULT.md")
    args = ap.parse_args(argv)
    r = pd.read_parquet(args.results)
    pr = r[r.run_id.str.startswith("s1_probe::")].copy()
    pr["model"] = pr.run_id.str.replace("s1_probe::", "", regex=False)
    pr["kind"] = pr["model"].map(kind_of)
    latest = pr.sort_values("value").drop_duplicates(["model", "metric"], keep="last")  # one row per cell
    piv = latest.pivot_table(index=["kind", "model"], columns="metric", values="value")
    cols = [c for c in ["probe_linear_period_norm", "probe_mlp_period_norm", "probe_linear_source", "probe_linear_genre_raw",
                        "probe_linear_provenance", "silhouette_period", "knn10_purity_period"] if c in piv.columns]
    within = [c for c in piv.columns if c.startswith("probe_linear_period_within_")]
    held = [c for c in piv.columns if c.startswith("probe_linear_period_heldout_")]
    L = ["# Representation sweep — S1 frozen vs S2 adapters vs S2 from-scratch", "",
         "Balanced accuracy unless noted; period chance ≈ .17 (6 classes ≥ 30 docs), source chance ≈ .17. "
         "A high SOURCE probe with a low WITHIN-source period probe means the model learned corpora, not time.", "",
         "## Main table", "",
         "| kind | model | " + " | ".join(c.replace("probe_linear_", "lin ").replace("probe_mlp_", "mlp ").replace("_norm", "").replace("_raw", "") for c in cols) + " |",
         "|---|---|" + "---|" * len(cols)]
    for (kind, model), row in piv.sort_index().iterrows():
        L.append(f"| {kind} | `{model}` | " + " | ".join("" if pd.isna(row.get(c)) else f"{row[c]:.3f}" for c in cols) + " |")
    if within or held:
        L += ["", "## Period probe within source / with a source held out (linear)", "",
              "| model | " + " | ".join(c.replace("probe_linear_period_", "") for c in within + held) + " |", "|---|" + "---|" * (len(within) + len(held))]
        for (kind, model), row in piv.sort_index().iterrows():
            L.append(f"| `{model}` | " + " | ".join("" if pd.isna(row.get(c)) else f"{row[c]:.3f}" for c in within + held) + " |")
    e2e = r[(r.run_id.str.startswith("ssl_e2e::")) & (r.metric == "quick_period_probe")].copy()
    if len(e2e):
        e2e["step"] = e2e["extra"].map(lambda e: json.loads(e).get("step")); e2e["params"] = e2e["extra"].map(lambda e: json.loads(e).get("params"))
        e2e["size"] = e2e["extra"].map(lambda e: json.loads(e).get("size")); e2e["objective"] = e2e["extra"].map(lambda e: json.loads(e).get("objective"))
        L += ["", "## From-scratch family — quick linear period probe during training", "", "| run | params | steps seen | first | best | last |", "|---|---|---|---|---|---|"]
        for run, g in e2e.groupby("run_id"):
            g = g.sort_values("step")
            L.append(f"| `{run}` | {g.params.iloc[0]/1e6:.1f} M | {int(g.step.max()):,} | {g.value.iloc[0]:.3f} | {g.value.max():.3f} | {g.value.iloc[-1]:.3f} |")
        fin = r[(r.run_id.str.startswith("ssl_e2e::")) & (r.metric == "final_loss")]
        if len(fin):
            L += ["", "| run | steps | hours | final loss |", "|---|---|---|---|"]
            for _, x in fin.iterrows():
                e = json.loads(x.extra); L.append(f"| `{x.run_id}` | {e.get('steps'):,} | {e.get('hours')} | {x.value:.4f} |")
    txt = "\n".join(L) + "\n"; os.makedirs(os.path.dirname(args.out), exist_ok=True); open(args.out, "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
