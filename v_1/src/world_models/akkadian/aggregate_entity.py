"""WB aggregation — roll the CELL-B entity probe JSONs into slide-ready tables.

Emits, next to the other akkadian summaries:
    results/summary_entity_best.csv    one row per (arm, dataset, site, rows) with the
                                       best-layer MC R2/rho, its sd, and the hold-out
    results/summary_entity_layerwise.csv  every layer, for the depth figure
    results/RESULTS_entity.md          the readable table the slide is built from

`rows=bare` is the paper-faithful probe (entity string alone); `rows=all` adds the
carrier sentences.

    python aggregate_entity.py
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(HERE, "results", "probes_entity")
OUT_DIR = os.path.join(HERE, "results")

# ladder order for the tables; anything unlisted is appended alphabetically
ARM_ORDER = ["llama2_70b", "llama2_13b", "llama2_7b", "gpt_oss_120b", "qwen3_32b",
             "qwen3_8b", "qwen3_1b7", "thalesian_cunei400m", "thalesian_akk300m",
             "umt5_base", "tfidf", "random", "llama2_7b_random",
             "llama2_13b_random", "llama2_70b_random"]
IS_CONTROL = {"tfidf", "random", "llama2_7b_random", "llama2_13b_random",
              "llama2_70b_random"}


def _rows_from(doc):
    for layer, per_tag in doc["layers"].items():
        for tag, sc in per_tag.items():
            base = {"arm": doc["method"], "entity_type": doc["entity_type"],
                    "site": doc["site"], "layer": int(layer), "rows": tag,
                    "n": sc.get("n"), "n_entities": sc.get("n_entities"),
                    "is_control": doc["method"] in IS_CONTROL}
            for probe in ("ridge_mc", "pls5_mc"):
                if probe in sc:
                    p = probe.replace("_mc", "")
                    base[f"{p}_mc_r2"] = sc[probe]["mc_r2"]
                    base[f"{p}_mc_r2_sd"] = sc[probe]["mc_r2_sd"]
                    base[f"{p}_mc_rho"] = sc[probe]["mc_rho"]
                    base[f"{p}_mc_rho_sd"] = sc[probe]["mc_rho_sd"]
            if "ridge_holdout" in sc:
                base["ridge_holdout_r2"] = sc["ridge_holdout"]["r2"]
                base["ridge_holdout_rho"] = sc["ridge_holdout"]["rho"]
            yield base


def main():
    docs = []
    for path in sorted(glob.glob(os.path.join(PROBE_DIR, "*", "*.json"))):
        with open(path) as f:
            docs.append(json.load(f))
    if not docs:
        raise SystemExit(f"no probe JSONs under {PROBE_DIR} — run WB2 first")

    lw = pd.DataFrame([r for d in docs for r in _rows_from(d)])
    lw.to_csv(os.path.join(OUT_DIR, "summary_entity_layerwise.csv"), index=False)

    # best layer per (arm, dataset, site, rows) by MC ridge R2
    best = (lw.sort_values("ridge_mc_r2", ascending=False)
              .groupby(["arm", "entity_type", "site", "rows"], as_index=False)
              .first())
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    best["_o"] = best.arm.map(lambda a: order.get(a, len(order)))
    best = best.sort_values(["entity_type", "site", "rows", "_o"]).drop(columns="_o")
    best.to_csv(os.path.join(OUT_DIR, "summary_entity_best.csv"), index=False)

    lines = ["# Cell B — entity-level results (obscure entities, English)", "",
             "Protocol: 200-draw Monte-Carlo over **entity-level** splits (20% of",
             "entities held out per draw); ridge, best layer. `bare` = the entity",
             "string alone (paper-faithful); `all` = plus five carrier sentences.",
             "A score witnesses learning only if it beats **both** the TF-IDF floor",
             "**and** the arm's own random-init twin.", ""]
    for et in sorted(best.entity_type.unique()):
        for rows in ("bare", "all"):
            sub = best[(best.entity_type == et) & (best.rows == rows)]
            if sub.empty:
                continue
            lines += [f"## {et} — rows={rows}", "",
                      "| arm | site | best layer | MC R2 | +/- | MC rho | +/- |",
                      "|---|---|--:|--:|--:|--:|--:|"]
            for _, r in sub.iterrows():
                name = f"*{r.arm}*" if r.is_control else r.arm
                lines.append(
                    f"| {name} | {r.site} | {r.layer} | {r.ridge_mc_r2:.3f} | "
                    f"{r.ridge_mc_r2_sd:.3f} | {r.ridge_mc_rho:.3f} | "
                    f"{r.ridge_mc_rho_sd:.3f} |")
            lines.append("")
    with open(os.path.join(OUT_DIR, "RESULTS_entity.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"[write] summary_entity_layerwise.csv ({len(lw)} rows)")
    print(f"[write] summary_entity_best.csv ({len(best)} rows)")
    print(f"[write] RESULTS_entity.md")


if __name__ == "__main__":
    main()
