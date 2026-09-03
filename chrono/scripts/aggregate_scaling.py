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
    latest = pr.drop_duplicates(["model", "metric"], keep="last")  # one row per cell: the LATEST run (append order), not the max
    piv = latest.pivot_table(index=["kind", "model"], columns="metric", values="value")
    # probe_linear_period_heldout_orcc is the one number that answers the
    # thesis question: train the period probe on every undated corpus, test it
    # on the dated royal inscriptions the SSL run never saw. It belongs in the
    # main table, next to the within-corpus number it so often contradicts.
    cols = [c for c in ["probe_linear_period_norm", "probe_mlp_period_norm", "probe_linear_period_heldout_orcc",
                        "probe_linear_source", "probe_linear_genre_raw",
                        "probe_linear_provenance", "silhouette_period", "knn10_purity_period"] if c in piv.columns]
    within = [c for c in piv.columns if c.startswith("probe_linear_period_within_")]
    held = [c for c in piv.columns if c.startswith("probe_linear_period_heldout_")]
    L = ["# Representation sweep — S1 frozen vs S2 adapters vs S2 from-scratch", "",
         "Balanced accuracy unless noted; period chance ≈ .17 (6 classes ≥ 30 docs), source chance ≈ .17. "
         "A high SOURCE probe with a low WITHIN-source period probe means the model learned corpora, not time.", "",
         "## How to read this (2026-09-03)", "",
         "**Within the SSL corpora the period is nearly free, and it is nearly the same question as the "
         "source.** Cells read the period at .81-.90 and the SOURCE at .92-.98, because here one implies "
         "the other (Old Babylonian = Archibab, Late Babylonian = the letters, Hellenistic = ORACC). The "
         "high period numbers, the k-NN purity and the UMAP silhouette are therefore not evidence that "
         "anything chronological was learned.", "",
         "**The `HELD-OUT dated` column is NOT usable; it is kept so the mistake stays on the record.** It "
         "scores balanced accuracy over the periods the dated royal inscriptions share with the undated "
         "pool — Neo-Assyrian (924 test documents), Middle Babylonian (28) and Hellenistic (ONE) — while "
         "the pool training the probe holds 52 Middle Babylonian and 5 Neo-Babylonian texts. Averaging "
         "three such classes is what produced \".10-.20, below chance\": an artefact of the class filter, "
         "not a finding about time. The 216 Neo-Babylonian inscriptions, the second largest group in the "
         "test set, were dropped from it entirely. The read-out that replaces it is C17 "
         "(`ssl/TRANSFER_DATED.md`): fit against an approximate period midpoint on the undated corpora, "
         "then Spearman against the true year of the dated inscriptions.", "",
         "## Main table", "",
         "| kind | model | " + " | ".join(c.replace("probe_linear_period_heldout_orcc", "HELD-OUT dated").replace("probe_linear_", "lin ").replace("probe_mlp_", "mlp ").replace("_norm", "").replace("_raw", "") for c in cols) + " |",
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
