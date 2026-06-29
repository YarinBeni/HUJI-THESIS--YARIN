"""J1 local sanity checks for the stress-test shared harness.

Run from repo root:
    python3 v_1/src/stress_tests/shared/build_and_validate.py

Validates (per the plan's J1 verification):
  * king-token locator coverage per ruler (word level, tier0);
  * gazetteer coverage of provenance rows (>=95% by count target);
  * anchors build for all rulers + a year grid;
  * proximity_error / great_circle metric sanity.
Writes results/j1_harness_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORCC = REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = HERE.parent / "results" / "j1_harness_report.json"

import king_token as kt
import anchors as anc
from metrics import proximity_error, great_circle_km


def main():
    df = pd.read_parquet(ORCC)
    report: dict = {}

    # --- king-token coverage ---
    sp = kt.load_spellings()
    rows, overall = kt.coverage_report(df, sp)
    report["king_token"] = {"per_ruler": rows, "overall": overall}

    # --- gazetteer coverage ---
    gaz = pd.read_csv(HERE / "sites_gazetteer.csv")
    known = set(str(x) for x in gaz["provenance"])
    prov = [str(x) for x in df["provenance"]]
    covered = sum(1 for x in prov if x in known)
    report["gazetteer"] = {
        "n_sites": int(len(gaz)),
        "rows_covered": int(covered),
        "rows_total": int(len(df)),
        "coverage_by_count": round(covered / len(df), 3),
        "uncovered_values": sorted(set(prov) - known - {"nan", "None"}),
    }

    # --- anchors ---
    ra = anc.build_ruler_anchors(df)
    ya = anc.build_year_anchors(df, step=10)
    report["anchors"] = {
        "n_ruler_anchors": len(ra),
        "n_year_anchors": len(ya),
        "ruler_anchor_example": ra[0] if ra else None,
        "year_anchor_example": ya[0] if ya else None,
    }

    # --- metric sanity ---
    yt = np.array([700, 680, 660, 640, 620], float)
    perfect = proximity_error(yt, yt)
    shuffled = proximity_error(yt, yt[::-1])
    gc = great_circle_km(36.359, 43.153, 32.542, 44.421)  # Nineveh -> Babylon ~ 430 km
    report["metric_sanity"] = {
        "proximity_perfect": round(perfect, 3),       # ~0.0
        "proximity_reversed": round(shuffled, 3),     # high
        "ninveh_babylon_km": round(gc, 1),            # ~420-440
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    o = overall
    print(f"king-token: {o['rows_with_name_found']}/{o['rows_with_mapped_ruler']} "
          f"= {o['coverage_within_mapped']} within mapped rulers "
          f"({o['share_of_corpus_mapped']} of corpus mapped)")
    print(f"gazetteer : {covered}/{len(df)} = {report['gazetteer']['coverage_by_count']} rows covered")
    print(f"anchors   : {len(ra)} ruler + {len(ya)} year")
    print(f"metrics   : proximity perfect={perfect:.3f} reversed={shuffled:.3f} "
          f"Nineveh-Babylon={gc:.1f}km")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
