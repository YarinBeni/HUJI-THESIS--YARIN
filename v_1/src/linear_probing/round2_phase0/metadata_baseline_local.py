# Run locally: python v_1/src/linear_probing/round2_phase0/metadata_baseline_local.py

"""B1 — metadata-only year baseline (the floor every embedding must beat).

Predicts year from metadata ALONE — one-hot {ruler, provenance, period}
(NOT genre, which is single-valued) — under the exact same Ridge
GroupKFold-by-ruler protocol as the embedding probes (Test 2), in both regimes:

  * imbalanced  : all 1,193 year-labeled fragments, 5-fold GroupKFold by ruler.
  * balanced : the same 200 MC draws (168 frags = 8 rulers x 21) used everywhere
               else, mean/std of per-draw Spearman/MAE/R2 over draws.

Because folds hold out whole rulers, the ruler one-hot is non-leaky (a held-out
ruler's column is all-zero at train time) — so this is an honest "date a ruler
you've never seen from find-site + period" floor, not a ruler lookup table.

Writes:
  results/orcc_round2_phase0/metadata_baseline_results.json   (full numbers)
  ../../geodesic/results/tables/T8_metadata_baseline.csv + .md (the table)
No GPU; runtime < 1 min.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "v_1/src/linear_probing"))
from pls_utils import fit_ridge_year_groupkfold  # noqa: E402

ORCC_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DRAWS_DIR    = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws"
OUT_JSON     = REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0/metadata_baseline_results.json"
TABLES_DIR   = REPO_ROOT / "v_1/src/geodesic/results/tables"

META_COLS = ["ruler", "provenance", "period"]   # genre is single-valued -> excluded
N_SPLITS  = 5


def one_hot(df):
    """One-hot the metadata columns of a (sub)frame -> float32 design matrix."""
    return pd.get_dummies(df[META_COLS].astype(str), dummy_na=True).values.astype(np.float32)


def main():
    orcc = pd.read_parquet(ORCC_PARQUET)
    labeled = orcc[orcc["year"].notna()].copy()
    labeled["fragment_id"] = labeled["fragment_id"].astype(str)
    by_id = labeled.set_index("fragment_id")
    print(f"Labeled ORCC: {len(labeled)}  rulers={labeled['ruler'].nunique()} "
          f"provenances={labeled['provenance'].nunique()} periods={labeled['period'].nunique()}",
          flush=True)

    out = {}

    # ---------------- imbalanced ----------------
    X = one_hot(labeled)
    y_raw = labeled["year"].values.astype(float)
    y_log = np.log(y_raw)
    groups = labeled["ruler"].astype(str).values
    full = fit_ridge_year_groupkfold(X, y_raw, y_log, groups, n_splits=N_SPLITS)
    out["imbalanced"] = full
    print(f"imbalanced  raw: sp={full['raw']['spearman_mean']:.3f} "
          f"mae={full['raw']['mae_mean']:.0f} r2={full['raw']['r2_mean']:.3f} "
          f"(valid_folds={full['raw']['n_valid_folds']}/{full['raw']['n_total_folds']})", flush=True)

    # ---------------- balanced MC ----------------
    draw_files = sorted(DRAWS_DIR.glob("draw_*.json"))
    print(f"balanced: {len(draw_files)} draws", flush=True)
    per_draw = {"raw": {"sp": [], "mae": [], "r2": []},
                "log": {"sp": [], "mae": [], "r2": []}}
    for df_path in draw_files:
        ids = json.load(open(df_path))["fragment_ids"]
        sub = by_id.loc[[i for i in ids if i in by_id.index]]
        Xb = one_hot(sub)
        yb_raw = sub["year"].values.astype(float)
        yb_log = np.log(yb_raw)
        gb = sub["ruler"].astype(str).values
        r = fit_ridge_year_groupkfold(Xb, yb_raw, yb_log, gb, n_splits=N_SPLITS)
        for yt in ("raw", "log"):
            per_draw[yt]["sp"].append(r[yt]["spearman_mean"])
            per_draw[yt]["mae"].append(r[yt]["mae_mean"])
            per_draw[yt]["r2"].append(r[yt]["r2_mean"])

    bal = {}
    for yt in ("raw", "log"):
        sp = np.array(per_draw[yt]["sp"], float)
        mae = np.array(per_draw[yt]["mae"], float)
        r2 = np.array(per_draw[yt]["r2"], float)
        bal[yt] = {
            "spearman_mean": float(np.nanmean(sp)), "spearman_std": float(np.nanstd(sp)),
            "mae_mean": float(np.nanmean(mae)), "mae_std": float(np.nanstd(mae)),
            "r2_mean": float(np.nanmean(r2)), "r2_std": float(np.nanstd(r2)),
            "n_draws": int(np.sum(~np.isnan(sp))),
        }
    out["balanced"] = bal
    print(f"balanced raw: sp={bal['raw']['spearman_mean']:.3f}±{bal['raw']['spearman_std']:.3f} "
          f"mae={bal['raw']['mae_mean']:.0f} r2={bal['raw']['r2_mean']:.3f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    # ---------------- T8 table ----------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    hdr = ["regime", "year_transform", "spearman_mean", "spearman_std",
           "mae_mean", "mae_std", "r2_mean", "r2_std", "n_valid_folds_or_draws"]
    rows = []
    for yt in ("raw", "log"):
        f = full[yt]
        rows.append(["imbalanced", yt, f["spearman_mean"], f["spearman_std"],
                     f["mae_mean"], f["mae_std"], f["r2_mean"], f["r2_std"], f["n_valid_folds"]])
    for yt in ("raw", "log"):
        b = bal[yt]
        rows.append(["balanced", yt, b["spearman_mean"], b["spearman_std"],
                     b["mae_mean"], b["mae_std"], b["r2_mean"], b["r2_std"], b["n_draws"]])
    with open(TABLES_DIR / "T8_metadata_baseline.csv", "w", newline="") as fh:
        import csv
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)

    (TABLES_DIR / "T8_metadata_baseline.md").write_text(f"""# Test 8 — Metadata-only year baseline (B1, the floor)

**What it is:** predicts year from **metadata alone** — one-hot {{ruler, provenance,
period}} (genre excluded, single-valued) — under the *same* Ridge GroupKFold-by-ruler
protocol as Test 2. Every embedding's year readout must beat this floor to claim it
learned anything beyond find-site + period bookkeeping.

**Non-leaky by construction:** folds hold out whole rulers, so the ruler one-hot is
all-zero at train time for the held-out ruler — the floor is "date an unseen ruler
from provenance + period", not a ruler->year lookup.

**Regimes:** `imbalanced` = all 1,193 labeled fragments; `balanced` = the same 200 MC
draws (168 frags, 8 rulers x 21) used by every other balanced result, mean/std over draws.

**Headline (year-raw):** balanced Spearman = {bal['raw']['spearman_mean']:.3f} ±
{bal['raw']['spearman_std']:.3f}, MAE = {bal['raw']['mae_mean']:.0f} yr.
Compare against T2 (Ridge) / T1 (PLS) balanced Spearman per model.
""")
    print(f"\nWrote {OUT_JSON} and T8_metadata_baseline.csv/.md", flush=True)


if __name__ == "__main__":
    main()
