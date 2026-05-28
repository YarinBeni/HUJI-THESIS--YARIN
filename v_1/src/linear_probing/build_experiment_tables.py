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
# qwen2.5 ("qwen") was dropped from the Round-3 write-up (2026-05-26): the model
# set is qwen3-only, with random = a random-init qwen3_8b. Excluded everywhere
# below, including the geodesic/LORO/Phase-D scoreboards where it was the former
# flagship (the headline manifold is now qwen3_1b7).
DROP_MODELS = {"qwen"}
MODELS = ["thalesian_cunei400m", "thalesian_akk300m", "mlm", "tfidf",
          "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]


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
                rows.append(["imbalanced", m, rec.get("cleaning"), rec.get("pooling"),
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
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         num(rec.get("mase_mean")), num(rec.get("mdape_mean")),
                         num(rec.get("shuffled_spearman_mean")),
                         num(rec.get("shuffled_r2_mean")),
                         rec.get("n_draws"), ""])
    # balanced_last MC (C3 sweep): same schema, configs key on '__last__'.
    # Source files only exist for the qwen3 family, thalesian x2, random.
    for m in MODELS:
        p = MC_DIR / f"{m}_pls__mc_balanced_last__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            _, cl, pl, lyr, tgt = key.split("__")
            rows.append(["balanced_last", m, cl, pl, lyr, tgt.replace("year-", ""), "",
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         num(rec.get("mase_mean")), num(rec.get("mdape_mean")),
                         num(rec.get("shuffled_spearman_mean")),
                         num(rec.get("shuffled_r2_mean")),
                         rec.get("n_draws"), ""])
    write_csv("T1_year_pls.csv", hdr, rows)
    write_md("T1_year_pls.md", """# Test 1 — Year regression, PLS

**What it is:** PLS (Partial Least Squares) finds the few directions in a model's
activation vectors that best predict the year, then linearly regresses year on them.
Supervised. "best layer" = the model layer whose activations predict year best.

**Data & split:** mean-pooled fragment activations. 5-fold **GroupKFold grouped by ruler**
(every fold tests on rulers it never trained on). `imbalanced` = all 1,193 year-labeled
fragments; `balanced` = 200 MC draws of 168 frags (8 rulers x 21), reported mean/std over draws.

**CSV `T1_year_pls.csv`** — one row per (regime, model, cleaning, pool, layer, year_transform, k).
Metrics: Spearman, R2, MAE (yr), MASE, MdAPE, and shuffled-label baselines (imbalanced only).
`k` = number of PLS components (imbalanced only). `n_valid_folds` < `n_total_folds` flags folds where
a held-out ruler spanned a single date (Spearman undefined) — common on the imbalanced full set,
which is why some imbalanced numbers are degenerate. Filter `year_transform=raw` for the headline.
""")


# ========================== TEST 2 — Year Ridge ============================
def test2_year_ridge():
    hdr = ["regime", "model", "cleaning", "pool", "layer", "year_transform",
           "spearman_mean", "spearman_std", "r2_mean", "r2_std",
           "mae_mean", "mae_std", "mase_mean", "mdape_mean",
           "shuffled_spearman_mean", "shuffled_r2_mean",
           "n_valid_folds_or_draws"]
    rows = []
    for m in MODELS:
        p = RIDGE_DIR / f"cls_numeric_results_{m}.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            rows.append(["imbalanced", m, rec.get("cleaning"), rec.get("pooling"),
                         rec.get("layer"), rec.get("year_transform"),
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         num(rec.get("mase_mean")), num(rec.get("mdape_mean")),
                         num(rec.get("shuffled_spearman_mean")),
                         num(rec.get("shuffled_r2_mean")),
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
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         num(rec.get("mase_mean")), num(rec.get("mdape_mean")),
                         num(rec.get("shuffled_spearman_mean")),
                         num(rec.get("shuffled_r2_mean")),
                         rec.get("n_draws")])
    # balanced_last (C3 sweep, last-token only)
    for m in MODELS:
        p = MC_DIR / f"{m}_cls_numeric__mc_balanced_last__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not (key.endswith("year-raw") or key.endswith("year-log")):
                continue
            _, cl, pl, lyr, tgt = key.split("__")
            rows.append(["balanced_last", m, cl, pl, lyr, tgt.replace("year-", ""),
                         num(rec.get("spearman_mean")), num(rec.get("spearman_std")),
                         num(rec.get("r2_mean")), num(rec.get("r2_std")),
                         num(rec.get("mae_mean")), num(rec.get("mae_std")),
                         num(rec.get("mase_mean")), num(rec.get("mdape_mean")),
                         num(rec.get("shuffled_spearman_mean")),
                         num(rec.get("shuffled_r2_mean")),
                         rec.get("n_draws")])
    write_csv("T2_year_ridge.csv", hdr, rows)
    write_md("T2_year_ridge.md", """# Test 2 — Year regression, Ridge

**What it is:** plain L2-penalized linear regression predicting year directly from the
activation vector — a single-direction readout, simpler than PLS.

**Data & split:** same as Test 1 (mean-pooled activations, 5-fold GroupKFold by ruler).
`imbalanced` Ridge was only run for the qwen3_* models; `balanced` exists for every model that
has a `*_cls_numeric` MC summary (mlm, tfidf, qwen3_*).

**CSV `T2_year_ridge.csv`** — one row per (regime, model, cleaning, pool, layer, year_transform).
Metrics: Spearman, R2, MAE, MASE, MdAPE, shuffled-Spearman, shuffled-R2. Headline =
`regime=balanced, year_transform=raw`. **N/A note:** MASE/MdAPE/shuffled-* are blank for older
Ridge runs (imbalanced qwen3_* and the existing balanced draws) — those columns only populate from
new draws produced by the widened `fit_ridge_year_groupkfold` (cluster jobs C4/C5).
""")


# ====================== TEST 3 — Ruler classification ======================
def _best_balanced_ruler_cfg(model, suffix="mc_balanced"):
    """Best balanced ruler config (max macro_f1_mean over __ruler keys).

    Prefer the CLS-logistic MC summary; fall back to the PLS-DA MC summary's
    ruler configs. `suffix` switches between `mc_balanced` and `mc_balanced_last`.
    Returns (cfg_key, rec, source) or (None, {}, None).
    """
    for src, fname in (("cls", f"{model}_cls__{suffix}__summary.json"),
                       ("pls", f"{model}_pls__{suffix}__summary.json")):
        p = MC_DIR / fname
        if not p.exists():
            continue
        pc = json.load(open(p)).get("per_config", {})
        ruler = {k: v for k, v in pc.items()
                 if k.endswith("__ruler") and "macro_f1_mean" in v}
        if not ruler:
            continue
        bk = max(ruler, key=lambda k: ruler[k]["macro_f1_mean"])
        return bk, ruler[bk], src
    return None, {}, None


def test3_ruler():
    p = RES / "orcc_round2_phase0/aggregated/phase0_summary.json"
    lb = json.load(open(p)).get("leaderboard_cls", [])
    lb_by_model = {e.get("method"): e for e in lb
                   if e.get("method") not in DROP_MODELS}
    hdr = ["model", "display", "cleaning", "pool",
           "r1_imbalanced_macro_f1", "r1_accuracy", "r1_best_layer",
           "balanced_mc_macro_f1_mean", "balanced_mc_macro_f1_std",
           "balanced_mc_macro_f1_median", "balanced_mc_layer", "n_draws",
           "balanced_accuracy_mean", "balanced_weighted_f1_mean",
           "balanced_chance_accuracy", "balanced_chance_macro_f1",
           "balanced_shuffled_accuracy_mean", "balanced_shuffled_macro_f1_mean",
           "balanced_source",
           "balanced_last_macro_f1_mean", "balanced_last_macro_f1_std",
           "balanced_last_accuracy_mean", "balanced_last_accuracy_std",
           "balanced_last_weighted_f1_mean",
           "balanced_last_chance_accuracy", "balanced_last_chance_macro_f1",
           "balanced_last_shuffled_accuracy_mean",
           "balanced_last_shuffled_macro_f1_mean",
           "balanced_last_layer", "balanced_last_n_draws",
           "balanced_last_source"]
    rows = []
    for m in MODELS:
        e = lb_by_model.get(m, {})
        r1, mc = e.get("r1") or {}, e.get("mc") or {}
        bk, brec, src = _best_balanced_ruler_cfg(m)
        # balanced_last: same logic, separate suffix.
        blk, blrec, blsrc = _best_balanced_ruler_cfg(m, suffix="mc_balanced_last")
        bl_layer = ""
        if blk:
            try:
                bl_layer = blk.split("__")[3]
            except Exception:
                bl_layer = ""
        rows.append([m, e.get("display"), e.get("cleaning"), e.get("pooling"),
                     num(r1.get("macro_f1")), num(r1.get("accuracy")), r1.get("best_layer"),
                     num(mc.get("macro_f1_mean")), num(mc.get("macro_f1_std")),
                     num(mc.get("macro_f1_median")), mc.get("layer"), mc.get("n_draws"),
                     num(brec.get("accuracy_mean")), num(brec.get("weighted_f1_mean")),
                     num(brec.get("chance_accuracy_mean")),
                     num(brec.get("chance_macro_f1_mean")),
                     num(brec.get("shuffled_accuracy_mean")),
                     num(brec.get("shuffled_macro_f1_mean")),
                     src or "",
                     num(blrec.get("macro_f1_mean")), num(blrec.get("macro_f1_std")),
                     num(blrec.get("accuracy_mean")), num(blrec.get("accuracy_std")),
                     num(blrec.get("weighted_f1_mean")),
                     num(blrec.get("chance_accuracy_mean")),
                     num(blrec.get("chance_macro_f1_mean")),
                     num(blrec.get("shuffled_accuracy_mean")),
                     num(blrec.get("shuffled_macro_f1_mean")),
                     bl_layer, blrec.get("n_draws") if blrec else "",
                     blsrc or ""])
    write_csv("T3_ruler_classification.csv", hdr, rows)
    write_md("T3_ruler_classification.md", """# Test 3 — Ruler classification

**What it is:** predict *which ruler* a fragment belongs to (multi-class). The "identify the
king" task — explicit names live here.

**Data & split:** 5-fold **StratifiedKFold** (same rulers in train & test). Metric = Macro-F1
(per-ruler F1 averaged equally; rare rulers count as much as common ones).

**Not apples-to-apples:** `r1_imbalanced` uses 11–41 rulers (chance tiny); `balanced_mc` uses
8 rulers (chance 0.125), so balanced Macro-F1 is mechanically higher. Use the columns to rank
*methods*, not to claim balancing "helped". CSV `T3_ruler_classification.csv` — one row per
**model** (iterating the full model set, not just the leaderboard) with both regimes side by side.

**Balanced full-ruler-set columns** (`balanced_*`) come from the best (max Macro-F1) `__ruler`
config in the model's balanced-MC summary — `balanced_source=cls` for the logistic readout,
`pls` if only PLS-DA exists. **N/A note:** `balanced_shuffled_accuracy_mean` /
`balanced_shuffled_macro_f1_mean` are blank when sourced from CLS-logistic — `fit_cls_cv` does not
compute a shuffled-label null (a principled N/A, not a gap); they populate only when the best ruler
config comes from PLS-DA, which does compute it.
""")


# ===================== TEST 3b — Ruler PLS-DA ==============================
def test3b_ruler_plsda():
    hdr = ["regime", "model", "cleaning", "pool", "layer", "k",
           "accuracy", "accuracy_std", "macro_f1", "macro_f1_std",
           "weighted_f1", "weighted_f1_std", "chance_accuracy", "chance_macro_f1",
           "shuffled_accuracy", "shuffled_macro_f1", "n_draws"]
    rows = []
    # Imbalanced: __ruler keys in pls_results_{m}.json (best_k_by_macro_f1).
    for m in MODELS:
        p = PLS_DIR / f"pls_results_{m}.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).items():
            if not key.endswith("__ruler") or "best_k_by_macro_f1" not in rec:
                continue
            bk = str(rec["best_k_by_macro_f1"])
            mm = rec.get("metrics_per_k", {}).get(bk, {})
            rows.append(["imbalanced", m, rec.get("cleaning"), rec.get("pooling"),
                         rec.get("layer"), bk,
                         num(mm.get("accuracy_mean")), num(mm.get("accuracy_std")),
                         num(mm.get("macro_f1_mean")), num(mm.get("macro_f1_std")),
                         num(mm.get("weighted_f1_mean")), num(mm.get("weighted_f1_std")),
                         num(mm.get("chance_accuracy")), num(mm.get("chance_macro_f1")),
                         num(mm.get("shuffled_accuracy_mean")),
                         num(mm.get("shuffled_macro_f1_mean")), ""])
    # Balanced: __ruler configs in {m}_pls__mc_balanced__summary.json.
    for m in MODELS:
        p = MC_DIR / f"{m}_pls__mc_balanced__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not key.endswith("__ruler") or "macro_f1_mean" not in rec:
                continue
            _, cl, pl, lyr, _ = key.split("__")
            rows.append(["balanced", m, cl, pl, lyr, "",
                         num(rec.get("accuracy_mean")), num(rec.get("accuracy_std")),
                         num(rec.get("macro_f1_mean")), num(rec.get("macro_f1_std")),
                         num(rec.get("weighted_f1_mean")), num(rec.get("weighted_f1_std")),
                         num(rec.get("chance_accuracy_mean")),
                         num(rec.get("chance_macro_f1_mean")),
                         num(rec.get("shuffled_accuracy_mean")),
                         num(rec.get("shuffled_macro_f1_mean")), rec.get("n_draws")])
    # balanced_last (C3 sweep): __ruler keys in *_pls__mc_balanced_last__summary.
    for m in MODELS:
        p = MC_DIR / f"{m}_pls__mc_balanced_last__summary.json"
        if not p.exists():
            continue
        for key, rec in json.load(open(p)).get("per_config", {}).items():
            if not key.endswith("__ruler") or "macro_f1_mean" not in rec:
                continue
            _, cl, pl, lyr, _ = key.split("__")
            rows.append(["balanced_last", m, cl, pl, lyr, "",
                         num(rec.get("accuracy_mean")), num(rec.get("accuracy_std")),
                         num(rec.get("macro_f1_mean")), num(rec.get("macro_f1_std")),
                         num(rec.get("weighted_f1_mean")), num(rec.get("weighted_f1_std")),
                         num(rec.get("chance_accuracy_mean")),
                         num(rec.get("chance_macro_f1_mean")),
                         num(rec.get("shuffled_accuracy_mean")),
                         num(rec.get("shuffled_macro_f1_mean")), rec.get("n_draws")])
    write_csv("T3b_ruler_plsda.csv", hdr, rows)
    write_md("T3b_ruler_plsda.md", """# Test 3b — Ruler classification, PLS-DA

**What it is:** the same "which ruler?" multi-class task as Test 3, but read out with **PLS-DA**
(PLS discriminant analysis — PLS regression onto one-hot ruler targets, argmax for the predicted
class) instead of logistic regression. A linear, low-rank discriminant readout; `k` = number of
PLS components.

**Data & split:** identical protocol to Test 3 — 5-fold **StratifiedKFold** (same rulers in train &
test). `imbalanced` = all labeled fragments (11-41 rulers; per-config best `k` chosen by Macro-F1);
`balanced` = 200 MC draws of 8 rulers x 21 frags, mean/std over draws.

**CSV `T3b_ruler_plsda.csv`** — one row per (regime, model, cleaning, pool, layer[, k]). Full ruler
metric set: accuracy, Macro-F1, weighted-F1, chance-accuracy, chance-Macro-F1, and shuffled-label
baselines (shuffled-acc, shuffled-Macro-F1). Imbalanced rows come straight from the `__ruler` keys
in `pls_results_{model}.json`; balanced rows from the `__ruler` configs in the
`{model}_pls__mc_balanced` summary. Same chance-rate caveat as Test 3 (8 vs 11-41 classes), so use
columns to rank methods, not to claim balancing helped.
""")


# ========================= TEST 4 — Geodesic ===============================
def test4_geodesic():
    hdr = ["regime", "method", "cleaning", "pool", "layer", "k_used",
           "isomap_spearman", "isomap_pairwise_acc",
           "isomap_neighbor_purity", "isomap_neighbor_sigma",
           "ebin_spearman", "ebin_pairwise_acc",
           "isomap_spearman_std", "isomap_pairwise_acc_std",
           "isomap_neighbor_purity_std", "isomap_neighbor_sigma_std",
           "n_draws"]
    rows = []
    # Imbalanced (existing scoreboard) — std/n_draws blank.
    d_imb = json.load(open(GEO / "geodesic_layer_scoreboard.json"))
    for r in d_imb:
        if r.get("method") in DROP_MODELS:
            continue
        rows.append(["imbalanced", r.get("method"), r.get("cleaning"),
                     r.get("pool"), r.get("layer"), r.get("k_used"),
                     num(r.get("isomap_spearman")), num(r.get("isomap_pairwise_acc")),
                     num(r.get("isomap_neighbor_purity")),
                     num(r.get("isomap_neighbor_sigma")),
                     num(r.get("ebin_spearman")), num(r.get("ebin_pairwise_acc")),
                     "", "", "", "", ""])
    # Balanced (new scoreboard) — ebin_* blank, std/n_draws populated.
    bp = GEO / "geodesic_layer_scoreboard_balanced.json"
    if bp.exists():
        for r in json.load(open(bp)):
            if r.get("method") in DROP_MODELS:
                continue
            rows.append(["balanced", r.get("method"), r.get("cleaning"),
                         r.get("pool"), r.get("layer"), r.get("k_used"),
                         num(r.get("isomap_spearman_mean")),
                         num(r.get("isomap_pairwise_acc_mean")),
                         num(r.get("isomap_neighbor_purity_mean")),
                         num(r.get("isomap_neighbor_sigma_mean")),
                         "", "",
                         num(r.get("isomap_spearman_std")),
                         num(r.get("isomap_pairwise_acc_std")),
                         num(r.get("isomap_neighbor_purity_std")),
                         num(r.get("isomap_neighbor_sigma_std")),
                         r.get("n_draws")])
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
    hdr = ["regime", "method", "cleaning", "pool", "layer", "pacc_full",
           "pacc_loro_mean", "drop", "n_rulers",
           "pacc_full_std", "pacc_loro_mean_std", "drop_std", "n_draws"]
    d = [r for r in d if r["method"] not in DROP_MODELS]
    rows = [["imbalanced", r["method"], r["cleaning"], r["pool"], r["layer"],
             num(r["pacc_full"]), num(r["pacc_loro_mean"]), num(r["drop"]),
             r["n_rulers"], "", "", "", ""] for r in d]
    # Balanced LORO (C11) — drop "qwen" (qwen2.5) via DROP_MODELS, no per-ruler.
    bp = GEO / "loro_robustness_balanced.json"
    if bp.exists():
        for r in json.load(open(bp)):
            if r.get("method") in DROP_MODELS:
                continue
            rows.append(["balanced", r["method"], r["cleaning"], r["pool"],
                         r["layer"],
                         num(r.get("pacc_full_mean")),
                         num(r.get("pacc_loro_mean_mean")),
                         num(r.get("drop_mean")),
                         "",
                         num(r.get("pacc_full_std")),
                         num(r.get("pacc_loro_mean_std")),
                         num(r.get("drop_std")),
                         r.get("n_draws")])
    write_csv("T5_loro.csv", hdr, rows)
    # per-ruler detail (imbalanced only — balanced has no per-ruler).
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
        if r.get("method") in DROP_MODELS:
            continue
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

**Metric harmonization (T7h):** the masking job (`tfidf_namemask_results.json`) only persisted the
headline triple per condition — year **Spearman**, year **MAE**, ruler **Macro-F1** (each with std
and n_draws). The rest of the unified year/ruler metric sets (R2, MASE, MdAPE, shuffled-* for year;
accuracy, weighted-F1, chance-*, shuffled-* for ruler) were **not computed by the masking job** and
are a principled **N/A** here, not a gap — re-running masking with the full metric set is out of
scope (the plan says "do not re-run any masking job"). The CSV therefore carries only the three
metrics the source actually provides.
""")


# ===================== TEST 9 — Direct elicitation =========================
def test9_elicitation():
    hdr = ["model", "variant", "headline_metric", "headline_value",
           "parse_errors", "n_total", "n_scoreable", "extra"]
    rows = []
    models = ["qwen3_1b7", "qwen3_8b", "qwen3_32b"]
    base = RES / "orcc_round2_phase1a"
    for m in models:
        for v in ("kp0", "kp1", "kp2"):
            p = base / f"direct_kp_{m}" / "scores" / f"{v}_metrics.json"
            if not p.exists():
                rows.append([m, v, "", "", "", "", "", "MISSING"])
                continue
            d = json.load(open(p))
            if v == "kp0":
                total = d.get("total")
                pe = d.get("parse_errors") or 0
                scoreable = (total - pe) if isinstance(total, int) else ""
                extra = (f"correct={d.get('correct')} "
                         f"error_rate={d.get('error_rate')}")
                rows.append([m, v, "accuracy_tol50yr",
                             num(d.get("accuracy")), pe, total, scoreable, extra])
            elif v == "kp1":
                extra = f"periods={d.get('total_periods')}"
                rows.append([m, v, "aggregate_recall",
                             num(d.get("aggregate_recall")),
                             d.get("parse_errors"), d.get("total_targets"),
                             d.get("total_hits"), extra])
            elif v == "kp2":
                extra = (f"declined_correctly={d.get('declined_correctly')} "
                         f"gate_pass={d.get('gate_pass')} "
                         f"threshold={d.get('gate_threshold')}")
                rows.append([m, v, "hallucination_rate",
                             num(d.get("hallucination_rate")),
                             d.get("parse_errors"), d.get("total"),
                             d.get("scoreable"), extra])
    write_csv("T9_elicitation.csv", hdr, rows)
    write_md("T9_elicitation.md", """# Test 9 — Direct elicitation (kp0 / kp1 / kp2)

**What it is:** *prompted* knowledge probes — does the LLM, asked plainly in
English, produce the correct king/date answer? Three variants:

- **kp0 — knows-reign-dates.** "When did <king> rule?" Scored as accuracy
  within a 50-year tolerance window.
- **kp1 — king -> date recall.** Given a historical period, can the model
  recall the kings that fit? Aggregate-recall = total_hits / total_targets.
- **kp2 — hallucination gate.** Given fabricated/uncertain names, does the
  model decline rather than confabulate? Headline = hallucination rate;
  gate passes if the rate falls below `gate_threshold`.

**Evaluation sizes are SMALL by design:** 8 questions per variant — these are
targeted king/period probes, not corpus-wide labels. Read the numbers as
sanity-check signal, not full benchmarks. The **PASS gate** is on kp2 only
(if the model can't suppress hallucinated dates it's a deal-breaker even when
kp0/kp1 look fine).

**CSV `T9_elicitation.csv`** — one row per (model, variant). Headline metric
varies per variant; `extra` carries auxiliary counters. Three models present:
qwen3_1b7, qwen3_8b, qwen3_32b.
""")


# ===================== TEST 10 — Prompted reprobe ==========================
def test10_prompt_reprobe():
    import re
    hdr = ["model", "variant", "pool", "layer", "task",
           "headline_metric", "headline_value", "std", "n_labeled", "extra"]
    rows = []
    pat = re.compile(r"pv(\d+)__(last|mean)__L(\d+)__(cls|pls)\.json$")
    for m in ("qwen3_1b7", "qwen3_8b", "qwen3_32b"):
        rdir = RES / f"orcc_round2_phase1b_{m}" / "reprobing"
        if not rdir.exists():
            continue
        for f in sorted(rdir.glob("pv*__*__L*__*.json")):
            mt = pat.search(f.name)
            if not mt:
                continue
            pv, pool, lyr, task_tag = mt.groups()
            d = json.load(open(f))
            if task_tag == "pls":
                yr = d.get("year_raw") or {}
                mpk = yr.get("metrics_per_k", {})
                bk = yr.get("best_k_by_spearman")
                bkr = mpk.get(str(bk), {}) if bk is not None else {}
                extra = (f"k={bk} r2={bkr.get('r2_mean')} "
                         f"mae={bkr.get('mae_mean')}")
                rows.append([m, f"pv{pv}", pool, f"L{lyr}", "year",
                             "spearman_mean_year_raw",
                             num(bkr.get("spearman_mean")),
                             num(bkr.get("spearman_std")),
                             d.get("n_groups") or d.get("n_labeled"),
                             extra])
            else:  # cls
                extra = (f"accuracy={d.get('accuracy_mean')} "
                         f"chance={d.get('chance_macro_f1')} "
                         f"weighted_f1={d.get('weighted_f1_mean')}")
                rows.append([m, f"pv{pv}", pool, f"L{lyr}", "ruler",
                             "macro_f1_mean",
                             num(d.get("macro_f1_mean")),
                             num(d.get("macro_f1_std")),
                             d.get("n_fragments"),
                             extra])
    write_csv("T10_prompt_reprobe.csv", hdr, rows)
    write_md("T10_prompt_reprobe.md", """# Test 10 — Prompted reprobe (pv0/pv1/pv2/pv3)

**What it is:** does *prompting* the model first (context, framing, few-shot)
shift the linear probes' headline? For each prompt variant we re-extract
activations under that prompt and re-run the standard year-PLS / ruler-CLS
probes. Variants:

- **pv0** — headline "context-only" probe (the Round-3 default; no system
  prompt; last-token-inside-fragment pooling).
- **pv1 / pv2 / pv3** — control variants (system prompt swaps, few-shot
  injection, format perturbations). See
  `v_1/src/linear_probing/results/orcc_round2_phase1b/prompts/APPROVED.md` for
  the locked text and hashes.

**Coverage gaps (read carefully):**

- **qwen3_1b7** — pv0 complete (5 layers x 2 pools x {cls,pls} = 20 files);
  pv1 only L00/last/{cls,pls} = 2 files; pv2 / pv3 absent.
- **qwen3_8b** — pv0 only; even within pv0, `mean` pooling only at L00.
  Total 12 files. No pv1-3.
- **qwen3_32b** — pv0-pv3 fully complete (5 layers x 2 pools x {cls,pls} x 4
  variants = 80 files) plus a `phase1b_summary.json` side file.

The **cross-model comparison should focus on pv0** (the only variant present
for all three). Treat pv1-3 as **32B-only sensitivity** runs.

**CSV `T10_prompt_reprobe.csv`** — one row per (model, variant, pool, layer,
task). For PLS year: best-k Spearman from `metrics_per_k`. For CLS ruler:
Macro-F1 from the file's top level.
""")


def main():
    print("Writing per-experiment tables to", OUT)
    test1_year_pls()
    test2_year_ridge()
    test3_ruler()
    test3b_ruler_plsda()
    test4_geodesic()
    test5_loro()
    test6_phase_d()
    test7_name_masking()
    test9_elicitation()
    test10_prompt_reprobe()
    # index
    write_md("README.md", """# Round 3 — per-experiment tables

Each test has a `.md` (what it is, data/split, how to read the CSV) and a `.csv` (every config x
every metric, straight from the result JSONs). Regenerate with
`python v_1/src/linear_probing/build_experiment_tables.py`.

| Test | MD | CSV(s) |
|---|---|---|
| 1 Year regression — PLS | T1_year_pls.md | T1_year_pls.csv |
| 2 Year regression — Ridge | T2_year_ridge.md | T2_year_ridge.csv |
| 3 Ruler classification — CLS (logistic) | T3_ruler_classification.md | T3_ruler_classification.csv |
| 3b Ruler classification — PLS-DA | T3b_ruler_plsda.md | T3b_ruler_plsda.csv |
| 4 Geodesic / Isomap manifold | T4_geodesic.md | T4_geodesic.csv |
| 5 LORO leave-one-ruler-out | T5_loro.md | T5_loro.csv, T5_loro_per_ruler.csv |
| 6 Phase D visualization | T6_phase_d.md | T6_phase_d.csv |
| 7 TF-IDF name-masking control | T7_name_masking.md | T7_name_masking.csv |
| 9 Direct elicitation (kp0/kp1/kp2) | T9_elicitation.md | T9_elicitation.csv |
| 10 Prompted reprobe (pv0-pv3) | T10_prompt_reprobe.md | T10_prompt_reprobe.csv |

See also `../RESULTS_BY_TEST.md` (narrative, best-config tables) and
`../EXPERIMENTS_SUMMARY.md` (advisor-facing, embedded plots).
""")
    print("Done.")


if __name__ == "__main__":
    main()
