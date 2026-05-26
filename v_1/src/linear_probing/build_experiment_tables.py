#!/usr/bin/env python3
"""Generate per-experiment CSV (every config x every metric) + MD explanation.

Reads the committed result JSONs and writes, into
`v_1/src/geodesic/results/tables/`, one CSV + one MD per experiment so results
are inspectable in a spreadsheet without re-reading prose. Run:

    python v_1/src/linear_probing/build_experiment_tables.py
"""
import json
import csv
import glob
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "v_1/src/linear_probing/results"
GEO = ROOT / "v_1/src/geodesic/results"
OUT = GEO / "tables"
OUT.mkdir(exist_ok=True)

PLS_DIR = RES / "orcc__probe_pls"
RIDGE_DIR = RES / "orcc__probe_cls_numeric"
MC_DIR = RES / "orcc_round2_phase0/probes"
MODELS = ["thalesian_cunei400m", "thalesian_akk300m", "mlm", "tfidf",
          "qwen", "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]


def num(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else ""


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def write_md(name, text):
    (OUT / name).write_text(text)


# ============================ TEST 1 — Year PLS =============================
def test1_year_pls():
    hdr = ["regime", "model", "cleaning", "pool", "layer", "year_transform", "k",
           "spearman_mean", "spearman_std", "r2_mean", "r2_std",
           "mae_mean", "mae_std", "mase_mean", "mdape_mean",
           "shuffled_spearman_mean", "shuffled_r2_mean",
           "n_valid_folds", "n_total_folds"]
    rows = []
    # full set: per-k expansion
    for m in MODELS:
        p = PLS_DIR / f"pls_results_{m}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        for key, rec in d.items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            yt = rec.get("year_transform")
            for k, mm in rec.get("metrics_per_k", {}).items():
                rows.append(["fullset", m, rec.get("cleaning"), rec.get("pooling"),
                             rec.get("layer"), yt, k,
                             num(mm.get("spearman_mean")), num(mm.get("spearman_std")),
                             num(mm.get("r2_mean")), num(mm.get("r2_std")),
                             num(mm.get("mae_mean")), num(mm.get("mae_std")),
                             num(mm.get("mase_mean")), num(mm.get("mdape_mean")),
                             num(mm.get("shuffled_spearman_mean")),
                             num(mm.get("shuffled_r2_mean")),
                             mm.get("n_valid_folds"), mm.get("n_total_folds")])
    # balanced MC: from *_pls summaries (year targets)
    for m in MODELS:
        p = MC_DIR / f"{m}_pls__mc_balanced__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            _, cl, pl, lyr, tgt = key.split("__")
            rows.append(["balanced", m, cl, pl, lyr, tgt.replace("year-", ""), "",
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         "", "", "", "", "", "",
                         rec.get("n_draws"), ""])
    write_csv("T1_year_pls.csv", hdr, rows)
    write_md("T1_year_pls.md", """# Test 1 — Year regression, PLS

**What it is:** PLS (Partial Least Squares) finds the few directions in a model's
activation vectors that best predict the year, then linearly regresses year on them.
Supervised. "best layer" = the model layer whose activations predict year best.

**Data & split:** mean-pooled fragment activations. 5-fold **GroupKFold grouped by ruler**
(every fold tests on rulers it never trained on). `fullset` = all 1,193 year-labeled
fragments; `balanced` = 200 MC draws of 168 frags (8 rulers x 21), reported mean/std over draws.

**CSV `T1_year_pls.csv`** — one row per (regime, model, cleaning, pool, layer, year_transform, k).
Metrics: Spearman, R2, MAE (yr), MASE, MdAPE, and shuffled-label baselines (fullset only).
`k` = number of PLS components (fullset only). `n_valid_folds` < `n_total_folds` flags folds where
a held-out ruler spanned a single date (Spearman undefined) — common on the imbalanced full set,
which is why some fullset numbers are degenerate. Filter `year_transform=raw` for the headline.
""")


# ========================== TEST 2 — Year Ridge ============================
def test2_year_ridge():
    hdr = ["regime", "model", "cleaning", "pool", "layer", "year_transform",
           "spearman_mean", "spearman_std", "r2_mean", "r2_std",
           "mae_mean", "mae_std", "n_valid_folds_or_draws"]
    rows = []
    for m in MODELS:
        p = RIDGE_DIR / f"cls_numeric_results_{m}.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            rows.append(["fullset", m, rec.get("cleaning"), rec.get("pooling"),
                         rec.get("layer"), rec.get("year_transform"),
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         rec.get("n_valid_folds")])
    for m in MODELS:
        p = MC_DIR / f"{m}_cls_numeric__mc_balanced__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            _, cl, pl, lyr, tgt = key.split("__")
            rows.append(["balanced", m, cl, pl, lyr, tgt.replace("year-", ""),
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         "", "", rec.get("n_draws")])
    write_csv("T2_year_ridge.csv", hdr, rows)
    write_md("T2_year_ridge.md", """# Test 2 — Year regression, Ridge

**What it is:** plain L2-penalized linear regression predicting year directly from the
activation vector — a single-direction readout, simpler than PLS.

**Data & split:** same as Test 1 (mean-pooled activations, 5-fold GroupKFold by ruler).
`fullset` Ridge was only run for the qwen3_* models; `balanced` exists for every model that
has a `*_cls_numeric` MC summary (mlm, tfidf, qwen, qwen3_*).

**CSV `T2_year_ridge.csv`** — one row per (regime, model, cleaning, pool, layer, year_transform).
Metrics: Spearman, R2, MAE. Headline = `regime=balanced, year_transform=raw`.
""")


# ====================== TEST 3 — Ruler classification ======================
def test3_ruler():
    p = RES / "orcc_round2_phase0/aggregated/phase0_summary.json"
    lb = json.load(open(p)).get("leaderboard_cls", [])
    hdr = ["model", "display", "cleaning", "pool",
           "r1_imbalanced_macro_f1", "r1_accuracy", "r1_best_layer",
           "balanced_mc_macro_f1_mean", "balanced_mc_macro_f1_std",
           "balanced_mc_macro_f1_median", "balanced_mc_layer", "n_draws"]
    rows = []
    for e in lb:
        r1, mc = e.get("r1") or {}, e.get("mc") or {}
        rows.append([e.get("method"), e.get("display"), e.get("cleaning"), e.get("pooling"),
                     num(r1.get("macro_f1")), num(r1.get("accuracy")), r1.get("best_layer"),
                     num(mc.get("macro_f1_mean")), num(mc.get("macro_f1_std")),
                     num(mc.get("macro_f1_median")), mc.get("layer"), mc.get("n_draws")])
    write_csv("T3_ruler_classification.csv", hdr, rows)
    write_md("T3_ruler_classification.md", """# Test 3 — Ruler classification

**What it is:** predict *which ruler* a fragment belongs to (multi-class). The "identify the
king" task — explicit names live here.

**Data & split:** 5-fold **StratifiedKFold** (same rulers in train & test). Metric = Macro-F1
(per-ruler F1 averaged equally; rare rulers count as much as common ones).

**Not apples-to-apples:** `r1_imbalanced` uses 11–41 rulers (chance tiny); `balanced_mc` uses
8 rulers (chance 0.125), so balanced Macro-F1 is mechanically higher. Use the columns to rank
*methods*, not to claim balancing "helped". CSV `T3_ruler_classification.csv` — one row per
(model, cleaning, pool) with both regimes side by side.
""")


# ========================= TEST 4 — Geodesic ===============================
def test4_geodesic():
    d = json.load(open(GEO / "geodesic_layer_scoreboard.json"))
    hdr = ["method", "cleaning", "pool", "layer", "k_used",
           "isomap_spearman", "isomap_pairwise_acc",
           "isomap_neighbor_purity", "isomap_neighbor_sigma",
           "ebin_spearman", "ebin_pairwise_acc"]
    rows = [[r.get("method"), r.get("cleaning"), r.get("pool"), r.get("layer"),
             r.get("k_used"),
             num(r.get("isomap_spearman")), num(r.get("isomap_pairwise_acc")),
             num(r.get("isomap_neighbor_purity")), num(r.get("isomap_neighbor_sigma")),
             num(r.get("ebin_spearman")), num(r.get("ebin_pairwise_acc"))] for r in d]
    write_csv("T4_geodesic.csv", hdr, rows)
    write_md("T4_geodesic.md", """# Test 4 — Geodesic / Isomap manifold (unsupervised)

**What it is:** rather than *training* a probe, ask whether fragments already lie along a curved
1-D "timeline" in activation space. **Isomap** builds a k-nearest-neighbor graph on the vectors
and "unrolls" it into one coordinate; years are never shown. `ebin` = an alternative
earliest-bin geodesic readout.

**Data & split:** unsupervised (no labels used to fit); evaluated on **all** fragments.
Metrics: **pairwise_acc (pacc)** = of fragment pairs >100yr apart, the fraction the 1-D
coordinate orders correctly (0.5 chance, 1.0 perfect) — the headline; **spearman** of the
coordinate vs year; **neighbor_purity** = fraction of each point's 10 nearest neighbors within
±100yr, with **neighbor_sigma** = σ above a shuffled-label null.

**CSV `T4_geodesic.csv`** — one row per (method, cleaning, pool, layer): all 728 swept configs.
Filter to max `isomap_pairwise_acc` per method for the best-layer leaderboard.
""")


# ========================= TEST 5 — LORO ===================================
def test5_loro():
    d = json.load(open(GEO / "loro_robustness.json"))
    hdr = ["method", "cleaning", "pool", "layer", "pacc_full",
           "pacc_loro_mean", "drop", "n_rulers"]
    rows = [[r["method"], r["cleaning"], r["pool"], r["layer"],
             num(r["pacc_full"]), num(r["pacc_loro_mean"]), num(r["drop"]),
             r["n_rulers"]] for r in d]
    write_csv("T5_loro.csv", hdr, rows)
    # per-ruler detail
    hdr2 = ["method", "cleaning", "pool", "layer", "ruler", "n",
            "pacc_loro", "pacc_cross", "drop"]
    rows2 = []
    for r in d:
        cfg = (r["method"], r["cleaning"], r["pool"], r["layer"])
        for pr in r.get("per_ruler", []):
            rows2.append([*cfg, pr["ruler"], pr["n"], num(pr["pacc_loro"]),
                          num(pr["pacc_cross"]), num(pr["drop"])])
    write_csv("T5_loro_per_ruler.csv", hdr2, rows2)
    write_md("T5_loro.md", """# Test 5 — LORO (Leave-One-Ruler-Out)

**What it is:** is the manifold a real *timeline*, or just "each ruler is its own blob near its
date"? Refit the Isomap manifold with **one ruler's fragments removed**, drop those held-out
fragments onto it, re-measure pacc. Small drop = genuine temporal axis.

**Data & split:** held out one ruler at a time (11 rulers); manifold fit on the other 10.
`drop` = pacc_full − mean(pacc over held-out rulers). STRONG if drop < 0.10.

**CSVs:** `T5_loro.csv` (one row per config, summary drop) and `T5_loro_per_ruler.csv`
(per held-out ruler: `pacc_loro` = held-in pacc, `pacc_cross` = held-out fragments' pacc, `n`).
""")


# ========================= TEST 6 — Phase D ================================
def test6_phase_d():
    hdr = ["method", "cleaning", "pool", "layer", "geodesic_spearman",
           "pairwise_order_acc", "arc_length_spearman", "n_bins",
           "bin_centers", "bin_counts"]
    rows = []
    for f in sorted(glob.glob(str(GEO / "phase_d/*_metrics.json"))):
        r = json.load(open(f))
        rows.append([r.get("method"), r.get("cleaning"), r.get("pool"), r.get("layer"),
                     num(r.get("geodesic_spearman")), num(r.get("pairwise_order_acc")),
                     num(r.get("arc_length_spearman")), r.get("n_bins"),
                     "|".join(map(str, r.get("bin_centers", []))),
                     "|".join(map(str, r.get("bin_counts", [])))])
    write_csv("T6_phase_d.csv", hdr, rows)
    write_md("T6_phase_d.md", """# Test 6 — Phase D visualization (centroid + spline)

**What it is:** a figure/quant check — bin fragments into 100-year windows, take each window's
3-D PCA centroid, fit a smooth spline through the centroids, and measure whether
distance-along-the-curve tracks century order.

**Data & split:** all fragments; 7 populated century bins. Metric = **arc_length_spearman**
(1.0 = the curve threads centuries in perfect order). `bin_centers`/`bin_counts` are `|`-joined.

**CSV `T6_phase_d.csv`** — one row per visualized config. 12 PNGs (4 colorings x 3 configs) live
in `../phase_d/` and are embedded in `../EXPERIMENTS_SUMMARY.md`.
""")


# ===================== TEST 7 — Name masking ===============================
def test7_name_masking():
    d = json.load(open(RES / "orcc_round2_phase0/tfidf_namemask_results.json"))
    hdr = ["cleaning", "condition", "year_spearman_mean", "year_spearman_std",
           "year_mae_mean", "year_mae_std", "ruler_macro_f1_mean",
           "ruler_macro_f1_std", "n_draws"]
    rows = []
    for key, rec in d.items():
        cl, cond = key.split("__")
        ysp, ymae, rf = rec["year_sp"], rec["year_mae"], rec["ruler_f1"]
        rows.append([cl, cond, num(ysp[0]), num(ysp[1]), num(ymae[0]), num(ymae[1]),
                     num(rf[0]), num(rf[1]), ysp[2]])
    write_csv("T7_name_masking.csv", hdr, rows)
    write_md("T7_name_masking.md", """# Test 7 — TF-IDF name-masking control

**What it is:** does a char-n-gram TF-IDF model date texts by reading the *king's name* or by
period spelling? We mask all personal names — `m-`/`f-` determinative tokens AND theophoric
`d-<god>-<predicate>` sentence-names (e.g. Nabu-kudurri-usur = Nebuchadnezzar -> `[PN]`), while
keeping bare god names — then re-date. Masking module: `../../linear_probing/name_masking.py`.

**Data & split:** balanced MC (200 draws x 168 frags). Year via Ridge GroupKFold-by-ruler ->
Spearman; ruler via logistic StratifiedKFold -> Macro-F1.

**CSV `T7_name_masking.csv`** — rows = {tier0,maximal} x {unmasked,masked}. Compare masked vs
unmasked within a cleaning: year Spearman is unchanged (dating != name lookup) while ruler
Macro-F1 drops (names did carry ruler identity).
""")


def main():
    print("Writing per-experiment tables to", OUT)
    test1_year_pls()
    test2_year_ridge()
    test3_ruler()
    test4_geodesic()
    test5_loro()
    test6_phase_d()
    test7_name_masking()
    # index
    write_md("README.md", """# Round 3 — per-experiment tables

Each test has a `.md` (what it is, data/split, how to read the CSV) and a `.csv` (every config x
every metric, straight from the result JSONs). Regenerate with
`python v_1/src/linear_probing/build_experiment_tables.py`.

| Test | MD | CSV(s) |
|---|---|---|
| 1 Year regression — PLS | T1_year_pls.md | T1_year_pls.csv |
| 2 Year regression — Ridge | T2_year_ridge.md | T2_year_ridge.csv |
| 3 Ruler classification | T3_ruler_classification.md | T3_ruler_classification.csv |
| 4 Geodesic / Isomap manifold | T4_geodesic.md | T4_geodesic.csv |
| 5 LORO leave-one-ruler-out | T5_loro.md | T5_loro.csv, T5_loro_per_ruler.csv |
| 6 Phase D visualization | T6_phase_d.md | T6_phase_d.csv |
| 7 TF-IDF name-masking control | T7_name_masking.md | T7_name_masking.csv |

See also `../RESULTS_BY_TEST.md` (narrative, best-config tables) and
`../EXPERIMENTS_SUMMARY.md` (advisor-facing, embedded plots).
""")
    print("Done.")


if __name__ == "__main__":
    main()
