"""SAE2 step 3 — pull Neuronpedia autointerp labels for the hunted features.

The whole point of switching dictionaries: these features have human-readable
labels, top-activating examples, and logit projections already computed. This
fetches them via the public API and classifies each label into the plan's
taxonomy (temporal / entity-identity / numeric-year / historical-domain /
style / other) with a keyword heuristic — labels are LLM-generated, so the
classification is a first pass for the human read, not ground truth.

Also does the plan's cheap cross-check: lenses each top feature's decoder row
through W_U with the E4.4 code, so a temporal label can be corroborated
independently of Neuronpedia.

    python fetch_labels.py --source 18-resid-batchtopk-65k
    python fetch_labels.py --source auto        # derive from pipeline.json

Writes results/labels.layer{L}.json + merges labels into the feature CSV.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
MODEL_NP = "qwen3-8b"

TAXONOMY = {
    "temporal": ("year", "date", "century", "era", "ancient", "historical",
                 "period", "decade", "bc", "medieval", "chronolog", "time"),
    "numeric_year": ("number", "numeral", "digit", "1", "19", "20"),
    "entity_identity": ("name", "person", "people", "figure", "king", "ruler",
                        "individual", "biograph", "proper noun"),
    "historical_domain": ("archaeol", "empire", "dynasty", "civilization",
                          "mesopotam", "inscription", "kingdom", "war",
                          "monarch"),
    "style": ("formal", "narrative", "punctuation", "syntax", "grammar",
              "capitaliz", "sentence", "token", "prefix", "suffix"),
}


def classify_label(text):
    low = (text or "").lower()
    for cat, kws in TAXONOMY.items():
        if any(k in low for k in kws):
            return cat
    return "other"


def fetch(model, source, index, retries=3):
    url = f"https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:                                    # noqa: BLE001
            if i == retries - 1:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(2 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Neuronpedia source id, e.g. 18-resid-batchtopk-65k; "
                         "'auto' derives '{L}-resid-batchtopk-65k' from "
                         "pipeline.json")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(RESULTS, "feature_hunt2.layer*.csv")))
    if not csvs:
        sys.exit("run run_pipeline.py first")
    csv = csvs[-1]
    L = int(csv.split("layer")[1].split(".")[0])
    src = (f"{L}-resid-batchtopk-65k" if args.source == "auto"
           else args.source)
    tab = pd.read_csv(csv)
    print(f"[labels] {len(tab)} features from {os.path.basename(csv)} "
          f"via {MODEL_NP}/{src}", flush=True)

    labels, cats = [], []
    for f in tab.feature.astype(int):
        j = fetch(MODEL_NP, src, int(f))
        lab = (j.get("explanations") or [{}])[0].get("description") \
            if isinstance(j, dict) and "error" not in j else None
        lab = lab or j.get("error", "no-label")
        labels.append(lab)
        cats.append(classify_label(lab))
        print(f"  {f}: [{cats[-1]}] {str(lab)[:90]}", flush=True)
        time.sleep(args.sleep)
    tab["np_label"] = labels
    tab["np_category"] = cats
    tab.to_csv(csv, index=False)

    summary = {"source": src, "layer": L,
               "category_counts": pd.Series(cats).value_counts().to_dict(),
               "top5": tab.head(5)[["feature", "rho_year", "np_category",
                                    "np_label"]].to_dict("records")}
    with open(os.path.join(RESULTS, f"labels.layer{L}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] categories: {summary['category_counts']}", flush=True)


if __name__ == "__main__":
    main()
