# Round 3 — Advisor-Facing Experiment Summary (Akkadian Cuneiform Interpretability)

**Audience:** thesis advisor. No prior jargon assumed — every metric is defined in plain
language at first use.
**Scope:** this document audits the *completed* Round 3 work. No new experiments were run;
all numbers below are read directly from the committed result JSON/CSV/PNG files. Where a
number could only be found in the prose report (and not in a standalone machine-readable
scoreboard), this is stated explicitly.

**Primary sources**
- Narrative report: `v_1/src/geodesic/results/orcc_round3_REPORT.md`
- Geodesic results: `v_1/src/geodesic/results/` (`phase_a_results.json`, `geodesic_layer_scoreboard.json`, `geodesic_best_layers.json`, `loro_robustness.json`, `phase_d/*_metrics.json`, `phase_d/*.png`)
- Balanced ruler-classification MC: `v_1/src/linear_probing/results/orcc_round2_phase0/aggregated/phase0_summary.json`
- Full-set PLS year-regression leaderboard: `v_1/src/linear_probing/results/orcc__probe_pls/pls_best_layers.json`
- Name-masking control: `v_1/src/linear_probing/results/orcc_round2_phase0/tfidf_namemask_results.json`

---

## Background in one paragraph

Akkadian cuneiform fragments in the ORCC corpus are labelled with the **year** (BCE) they
were written and the **ruler** under whom they were written. Round 2 had shown that a
domain-finetuned Akkadian encoder (Thalesian cuneiBase-400m) carries real chronological
signal. Round 3 asks two questions: (1) **Is that chronological signal organized as a smooth
geometric "manifold" (a curved 1-dimensional timeline) in the model's internal representation
space?** — tested unsupervised with Isomap/geodesic readout (Phases A–D). (2) **Does making
the language model bigger (Qwen3 1.7B → 8B → 32B) buy better dating accuracy once we remove
the corpus's ruler-imbalance?** — tested with a class-balanced Monte-Carlo probing sweep
(Phase E1) plus a TF-IDF name-masking control.

---

## Glossary of every metric used

| Term | Plain-language definition |
|---|---|
| **Spearman (Sp)** | Rank correlation between predicted and true year, in [−1, 1]. 1 = perfect ordering by date; 0 = no relationship. Insensitive to the exact scale, only to ordering. |
| **MAE (years)** | Mean Absolute Error — average size of the dating mistake, in years. Lower is better. |
| **R²** | Fraction of year-variance explained. 1 = perfect; 0 = no better than predicting the mean; negative = worse than the mean. |
| **Macro-F1** | Average per-ruler classification quality (precision/recall harmonic mean), averaged over rulers so rare rulers count as much as common ones. 0–1, higher better. Chance ≈ 0.026 for the imbalanced set. |
| **pacc (pairwise-order accuracy, ±100yr margin)** | Take every pair of fragments more than 100 years apart in true date. Fraction of those pairs the model orders correctly along its 1D readout. 0.5 = coin flip, 1.0 = perfect chronological ordering. The headline geodesic metric. |
| **Neighbor purity (k=10, ±100yr)** | For each fragment, look at its 10 nearest neighbors in embedding space; purity = fraction within ±100 years. Measures *local* temporal clustering. Reported against a shuffled-label null (how many σ above chance). |
| **Geodesic Spearman** | Spearman between true year and the fragment's position along the unsupervised Isomap 1D coordinate (the "unrolled" manifold axis). |
| **Arc-length Spearman** | After binning fragments into 100-year windows and fitting a smooth 3D curve (spline) through the bin centroids, Spearman between distance-along-the-curve and the bin's year. 1.0 = the curve threads the centuries in perfect chronological order. |
| **LORO drop** | Leave-One-Ruler-Out drop. Refit the manifold without one ruler's fragments, then place those held-out fragments on it; drop = how much pacc falls. Small drop = the timeline is genuinely temporal, not just "each ruler is its own blob." |

**Cleaning tiers:** `tier0` = light normalization; `maximal` = aggressive normalization.
**Pooling:** `mean` = average the per-token vectors of a fragment; `last` = use only the
last token's vector.
**Models:** `qwen` = Qwen2.5-7B (no Akkadian training); `qwen3_1b7/8b/32b` = Qwen3 scale
sweep (no Akkadian training); `thalesian_cunei400m` / `thalesian_akk300m` = Akkadian-
finetuned MLM encoders; `mlm` = small Akkadian MLM (Aeneas); `tfidf` = character n-gram
baseline (spelling only, no semantics); `random` = random-initialized Qwen control.

**Corpus (canonical):** ORCC parquet, 1,202 fragments total, **1,193 year-labeled**
(`phase_0_inventory.json`). 11 rulers. The corpus is heavily Neo-Assyrian-dominated
(≈1,137 / 1,192 fragments fall in 550–750 BCE).

---

## Phase A — Single-layer geodesic proof-of-concept

**Data:** all 1,193 year-labeled ORCC fragments, no balancing.
**Target:** temporal ordering / year (unsupervised, no labels used to fit).
**Method:** thalesian_cunei400m, tier0, mean pooling, **Layer 12** (the layer Round-2 PLS had
picked as best). Pipeline = StandardScaler → PCA-64 → L2-normalize → smallest-connected kNN
graph → **Isomap 1D** embedding. Compared against a supervised **PLS** (Partial Least Squares)
reference refit on the same activations.
**Source:** `phase_a_results.json`.

| Readout | Spearman | pacc (±100yr) | Neighbor purity (k=10) | σ above null |
|---|---|---|---|---|
| PLS reference (full-set, year_raw) | 0.467 | — | — | — |
| PLS reference (refit on these activations) | 0.633 | 0.859 | — | — |
| **Isomap 1D (unsupervised)** | **0.035** | **0.547** | **0.796** | **+13.39σ** |
| Isomap, earliest-bin anchored variant | 0.002 | 0.440 | 0.778 | +6.77σ |

**Gate verdict: FAIL** — Isomap pacc 0.547 < 0.60 threshold; recorded verdict in JSON:
"NULL → round 3 negative result; proceed to Phase 2 (scale)."

**Takeaway:** at the PLS-best layer the temporal signal is *locally* real (neighbor purity
+13σ above chance) but does **not** unroll into a single globally-monotone 1D timeline — which
motivated sweeping all layers in Phase B rather than trusting the PLS-selected layer.

---

## Phase B — Full layer × method × pooling geodesic scoreboard

**Data:** all 1,193 year-labeled fragments, no balancing.
**Target:** unsupervised temporal ordering (pacc) and geodesic Spearman.
**Method:** same Isomap pipeline as Phase A, run over every (model, cleaning, pooling) combo
and **every available layer**; the best layer per combo is reported.
**Source:** `geodesic_best_layers.json` (per-combo best layer) and the 353KB
`geodesic_layer_scoreboard.json` (per-layer detail).

Full scoreboard (sorted by best Isomap pacc), exact numbers from `geodesic_best_layers.json`:

| Model | Cleaning | Pool | Best L | Isomap pacc | Isomap Sp | Earliest-bin pacc | Earliest-bin Sp |
|---|---|---|---|---|---|---|---|
| **qwen** | maximal | mean | **1** | **0.7307** | 0.3324 | 0.6176 | 0.2003 |
| qwen3_1b7 | tier0 | mean | 1 | 0.7233 | 0.2496 | 0.5777 | 0.1453 |
| qwen3_32b | maximal | mean | 1 | 0.7164 | 0.3101 | 0.5432 | 0.1180 |
| qwen3_8b | maximal | mean | 1 | 0.7157 | 0.3158 | 0.6454 | 0.2476 |
| qwen3_1b7 | maximal | mean | 1 | 0.7132 | 0.2992 | 0.6232 | 0.2203 |
| qwen | tier0 | mean | 7 | 0.7119 | 0.2248 | 0.4933 | 0.0226 |
| qwen3_8b | tier0 | mean | 2 | 0.7034 | 0.2793 | 0.5588 | 0.1274 |
| qwen3_32b | tier0 | mean | 1 | 0.6877 | 0.2138 | 0.4880 | 0.0424 |
| **thalesian_cunei400m** | maximal | mean | **7** | **0.6806** | 0.2426 | 0.6195 | 0.0706 |
| thalesian_akk300m | tier0 | mean | 0 | 0.6619 | 0.1852 | 0.5549 | 0.0833 |
| thalesian_akk300m | maximal | mean | 3 | 0.6614 | 0.2631 | 0.5336 | 0.0203 |
| thalesian_akk300m | tier0 | last | 1 | 0.6555 | 0.1861 | 0.4615 | 0.0055 |
| **thalesian_cunei400m** | tier0 | mean | **6** | **0.6453** | 0.1082 | 0.5633 | 0.0940 |
| thalesian_akk300m | maximal | last | 1 | 0.6271 | 0.1443 | 0.6060 | 0.1488 |
| qwen3_32b | tier0 | last | 6 | 0.6153 | 0.1413 | 0.5648 | 0.0863 |
| qwen | tier0 | last | 6 | 0.6152 | 0.1344 | 0.5552 | 0.0815 |
| qwen3_8b | tier0 | last | 10 | 0.6046 | 0.1326 | 0.5401 | 0.0506 |
| thalesian_cunei400m | maximal | last | 4 | 0.5987 | 0.0828 | 0.5181 | 0.0026 |
| qwen | maximal | last | 27 | 0.5981 | 0.1416 | 0.5171 | 0.0428 |
| qwen3_32b | maximal | last | 38 | 0.5909 | 0.0862 | 0.5030 | 0.0086 |
| qwen3_1b7 | tier0 | last | 3 | 0.5895 | 0.1119 | 0.5503 | 0.0640 |
| qwen3_8b | maximal | last | 13 | 0.5846 | 0.0665 | 0.4650 | 0.0065 |
| qwen3_1b7 | maximal | last | 5 | 0.5803 | 0.0425 | 0.5325 | 0.0651 |
| thalesian_cunei400m | tier0 | last | 1 | 0.5734 | 0.0550 | 0.5270 | 0.0279 |

Key facts confirmed by the numbers: (1) **mean pooling beats last-token across the board**;
(2) **qwen (Qwen2.5-7B, no Akkadian training) has the single best manifold (pacc 0.731 at L1)**,
above domain-finetuned Thalesian (0.681 at L7); (3) **Layer 1 dominates** for all Qwen-family
mean-pool configs; (4) the three Qwen3 sizes (1b7 0.713/0.723, 8b 0.716, 32b 0.716/0.688) are
within noise — **no geodesic scale effect**; (5) Phase B gate **PASSES** for Thalesian
maximal/mean/L7 (0.681 ≥ 0.60).

**Takeaway:** an untrained model's first (token-embedding) layer already lays out fragments
along a clean chronological curve — consistent with lexical/orthographic drift over centuries —
and bigger models do not improve this geometry.

---

## Phase C — LORO honesty pass (leave-one-ruler-out)

**Data:** the three top configs from Phase B; all 1,193 fragments, held out one ruler at a time
(11 rulers).
**Target:** is the manifold genuinely a *timeline*, or just one cluster per ruler?
**Method:** refit PCA+Isomap on held-in rulers only, project the held-out ruler with
`Isomap.transform()`, measure cross-ruler pacc; report the drop vs the full-fit pacc.
Gate: mean drop < 0.10 = STRONG; < 0.20 = HEDGED; ≥ 0.20 = WEAK.
**Source:** `loro_robustness.json`.

| Config | Best L | pacc_full | pacc_loro_mean | Mean drop | Verdict |
|---|---|---|---|---|---|
| qwen / maximal / mean | 1 | 0.7307 | 0.7228 | **0.0079** | **STRONG** |
| thalesian_cunei400m / maximal / mean | 7 | 0.6806 | 0.6255 | **0.0550** | **STRONG** |
| thalesian_cunei400m / tier0 / mean | 6 | 0.6453 | 0.6166 | **0.0288** | **STRONG** |

Notable per-ruler drops (from JSON):

| Config | Ruler | n | per-ruler drop | Note |
|---|---|---|---|---|
| thalesian maximal/L7 | Nabonidus | 68 | 0.116 | only individual ruler over 0.10; era overlaps Nebuchadnezzar II |
| thalesian tier0/L6 | Nabopolassar | 15 | 0.216 | small-sample, near Isomap stability limit |
| all configs | Nebuchadnezzar I | 10 | negative / high-variance | temporally isolated (~1100 BCE) outlier; tiny n |
| qwen maximal/L1 | (max ruler) | — | ≤0.029 | near-perfect stability everywhere |

**Takeaway:** all three manifolds pass STRONG — the chronological ordering survives removing
any single ruler, so it is a real timeline rather than an artifact of per-ruler clusters. qwen's
L1 manifold is almost perfectly ruler-invariant (drop 0.008).

---

## Phase D — Centroid + spline visualization

**Data:** same three configs; fragments binned into 100-year windows (min 5 per bin → 7
populated bins).
**Target:** visualize and quantify whether century-centroids lie in chronological order along a
smooth curve.
**Method:** per bin, compute PCA-3D centroid; fit a √n-weighted cubic UnivariateSpline; measure
arc-length Spearman vs bin year. 12 PNGs (4 colorings × 3 configs).
**Source:** `phase_d/*_metrics.json` and `phase_d/*.png`.

| Config | Layer | Geodesic Sp | pacc | **Arc-length Sp** | # bins | Bin centers (BCE) | Bin counts |
|---|---|---|---|---|---|---|---|
| qwen / maximal / mean | 1 | 0.3324 | 0.7307 | **1.000** | 7 | 50, 550, 650, 750, 950, 1050, 1150 | 14, 171, 728, 238, 10, 18, 13 |
| thalesian_cunei400m / maximal / mean | 7 | 0.2426 | 0.6806 | **1.000** | 7 | (same centers) | (same counts) |
| thalesian_cunei400m / tier0 / mean | 6 | 0.1082 | 0.6453 | **1.000** | 7 | (same centers) | (same counts) |

**Arc-length Spearman = 1.0 for all three** — the spline threads all 7 centuries in perfect
chronological order despite extreme imbalance (728 of the ~992 binned fragments sit in the
650-BCE bin).

**Plot descriptions (all 12 PNGs viewed):** every plot is a 3D PCA scatter (PC1/PC2/PC3) of all
fragments, overlaid with black-diamond century centroids and a red cubic spline connecting them;
footer reads "Geodesic Spearman / pacc / arc-len Sp=1.000".
- *qwen L1, year coloring:* dense central cloud colored by year (purple-early → yellow-late);
  the red spline shoots from the cloud out to the two isolated early-century centroids. Visually
  the bulk is a tight blob (the 550–750 BCE mass) with a thin tail to older outliers.
- *qwen L1, geodesic coloring:* same geometry colored by the Isomap 1D coordinate (−1..+1) —
  color gradient aligns with the spline direction, confirming the unsupervised axis tracks the
  layout.
- *qwen L1, ruler coloring:* ~35 discrete category colors intermixed within the central blob —
  rulers are *not* cleanly separated into their own regions (consistent with the STRONG LORO).
- *qwen L1, archive coloring:* effectively single-valued colorbar (−0.1..0.1) → archive metadata
  is near-constant / unusable as a coloring; plot is monochrome.
- *thalesian L7 & L6, year coloring:* larger, more spread green/teal clouds (more populated PCA
  spread than qwen); centroids + spline again ordered by century. L6 (tier0) is the most diffuse.

**Readability caveat (also flagged in the report):** the 550–750 BCE over-representation makes
all scatters a dense blob; the arc-length=1.0 result rests on only 7 century centroids, several
of which (50, 950, 1050, 1150 BCE) have 10–18 fragments each.

**Takeaway:** the strongest possible visualization outcome — century centroids are perfectly
chronologically ordered along a smooth 3D curve for all three configs.

---

## Phase E1 — Qwen3 scale sweep (year regression)

This phase has two regression readouts (PLS and Ridge) reported both on the **full imbalanced
set** and under **class-balanced Monte-Carlo (MC)**.

### Balancing scheme (Balanced MC)
200 random draws, each draw = 168 fragments = **8 rulers × 21 fragments each**, to neutralize
the Neo-Assyrian ruler imbalance. Results reported as mean ± std across the 200 draws.

### E1a — Full-set PLS year-regression (no balancing)

**Data:** 1,193 year-labeled fragments. **Target:** year (raw BCE). **Method:** PLS-k regression,
best layer per model, maximal/mean. **Metric:** Spearman, MAE.
**Source for Thalesian/qwen/akk300m:** `orcc__probe_pls/pls_best_layers.json` (year-raw entries,
exact values below). **Source for qwen3_*:** report tables (the per-layer
`pls_results_qwen3_*.json` files exist but there is no consolidated best-layer scoreboard for
the qwen3 models; the report's best-layer figures are reproduced and labelled REPORT-only).

| Model | Best L | PLS Sp | MAE (yr) | R² | Source |
|---|---|---|---|---|---|
| qwen3_32b | 26 | 0.511 | 74.7 | — | report table (REPORT-only) |
| qwen3_1b7 | 6 | 0.484 | 75.1 | — | report table (REPORT-only) |
| qwen3_8b | 26 | 0.482 | 77.8 | — | report table (REPORT-only) |
| **thalesian_cunei400m** | **12** | **0.4670** | **75.10** | **0.1055** | pls_best_layers.json (`tier0__mean__year-raw`) |
| thalesian_cunei400m | 9 (maximal/mean) | 0.4166 | 77.06 | 0.1419 | pls_best_layers.json |
| thalesian_akk300m | 7 | 0.4346 | 76.54 | 0.0690 | pls_best_layers.json (`tier0__mean__year-raw`) |
| qwen (Qwen2.5-7B) | 5 | 0.1208 | 128.34 | −198.2 | pls_best_layers.json (`tier0__mean__year-raw`) |
| random (control) | 12 | 0.1843 | 127.87 | −175.7 | pls_best_layers.json |
| mlm (Aeneas) | 2 | −0.1154 | 139.51 | −152.9 | pls_best_layers.json (PLS year-raw is null for mlm) |

(Note: the report headlines Thalesian's 0.467 at tier0/L12; the maximal/mean best layer is L9 at
0.417. Both are in the JSON; the report uses the tier0 figure as the baseline.)

### E1b — Full-set Ridge (cls_numeric) year-regression

**Method:** Ridge regression on the cls_numeric activations, best layer per model.
**Source:** report tables (per-layer `orcc__probe_cls_numeric/cls_numeric_results_qwen3_*.json`
exist but no consolidated scoreboard; values are REPORT-only).

| Model | Best L | Ridge Sp | MAE (yr) |
|---|---|---|---|
| qwen3_1b7 | 2 | 0.444 | 80.6 |
| qwen3_8b | 2 | 0.439 | 81.7 |
| qwen3_32b | 62 | 0.429 | 84.5 |

### E1c — Balanced-MC PLS year-regression (the headline correction)

**Data:** 200 balanced draws (8×21). **Target:** year. **Method:** PLS-k, best layer per model.
**Metric:** Spearman mean ± std over 200 draws.
**Source:** **report tables only.** There is no standalone balanced-MC *year-regression*
scoreboard JSON in the audited directories (`orcc_round2_phase0/aggregated/phase0_summary.json`
contains only the balanced ruler-*classification* Macro-F1 numbers, see Phase-0 section below).
These figures could not be cross-checked against a raw JSON and are reproduced as REPORT-only.

| Model | Layer | Balanced PLS Sp ± std | Full-set Sp (ref) |
|---|---|---|---|
| **thalesian_cunei400m** | L12 | **0.411 ± 0.064** | 0.467 |
| qwen3_32b | L09 | 0.399 ± 0.063 | 0.511 |
| qwen3_1b7 | L09 | 0.371 ± 0.081 | 0.484 |
| qwen3_8b | L01 | 0.365 ± 0.068 | 0.482 |
| thalesian_akk300m | L06 | 0.344 ± 0.062 | 0.435 |

### E1d — Balanced-MC Ridge (cls_numeric) year-regression

**Source:** report tables only (REPORT-only; not in an audited JSON). The TF-IDF tier0 figure
(0.355 ± 0.069) is independently confirmed by `tfidf_namemask_results.json` (see control below).

| Model | Layer | Balanced Ridge Sp ± std |
|---|---|---|
| **mlm** (small Akkadian MLM) | L01 | **0.408 ± 0.061** |
| tfidf (char n-grams) | L00 | 0.355 ± 0.069 |
| qwen3_1b7 | L00 | 0.352 ± 0.068 |
| qwen3_8b | L01 | 0.332 ± 0.072 |
| qwen (Qwen2.5-7B) | L03 | 0.327 ± 0.069 |
| **qwen3_32b** | L06 | **0.326 ± 0.069** (last) |

**Takeaway:** the full-set "qwen3_32b wins (0.511)" disappears under balancing — Thalesian
(0.411) and qwen3_32b (0.399) become a statistical tie (all CIs ±0.06–0.08 overlap). On balanced
Ridge the ordering *inverts*: a small Akkadian MLM (0.408) and char n-gram TF-IDF (0.355) beat
every Qwen3, and the 32B model is dead last. Scale does not help dating once imbalance is removed.

---

## Phase 0 — Balanced-MC ruler classification (supporting data)

**Data:** 200 balanced draws (8×21). **Target:** ruler (11-way classification).
**Method:** logistic-regression CLS probe and PLS-DA probe, best layer per (method, cleaning,
pooling). **Metric:** Macro-F1, mean ± std over 200 draws.
**Source:** `orcc_round2_phase0/aggregated/phase0_summary.json` (and `phase0_report.md`). Round-1
(R1) column = imbalanced single-fit Macro-F1.

**CLS regime (direct ruler classification) — balanced MC, where available:**

| Method | Cleaning | Pool | MC L | R1 Macro-F1 | Balanced MC Macro-F1 ± std | Gate |
|---|---|---|---|---|---|---|
| **TF-IDF** | tier0 | na | 0 | 0.326 | **0.6496 ± 0.0368** | PASS |
| TF-IDF | maximal | na | 0 | 0.228 | 0.4980 ± 0.0403 | PASS |
| MLM (Aeneas) | tier0 | mean | 15 | 0.220 | 0.4604 ± 0.0435 | PASS |
| Thalesian cuneiBase-400m | tier0 | mean | 12 | 0.210 | 0.4479 ± 0.0432 | PASS |
| Qwen2.5-7B | tier0 | mean | 0 | 0.117 | 0.3521 ± 0.0417 | PASS |
| Thalesian AKK_300m | tier0 | mean | 8 | 0.160 | 0.3233 ± 0.0388 | PASS |

(random-Qwen and several Thalesian last-token / maximal configs had no MC entry — listed "n/a"
in `phase0_report.md`. Overall verdict in the JSON: INDETERMINATE, only because the random
control was never MC-probed.)

**PLS regime (ruler via year-PLS-DA) — balanced MC:**

| Method | Cleaning | Pool | MC L | Balanced MC Macro-F1 ± std |
|---|---|---|---|---|
| TF-IDF | tier0 | na | 0 | 0.4796 ± 0.0368 |
| MLM (Aeneas) | tier0 | mean | 14 | 0.3946 ± 0.0423 |
| TF-IDF | maximal | na | 0 | 0.3945 ± 0.0329 |
| Thalesian cuneiBase-400m | tier0 | mean | 11 | 0.3927 ± 0.0405 |
| Qwen2.5-7B | tier0 | mean | 3 | 0.3632 ± 0.0417 |
| Thalesian AKK_300m | tier0 | mean | 3 | 0.3457 ± 0.0391 |

**Takeaway:** for the *ruler* task too, the spelling-only TF-IDF baseline is the strongest
balanced model (0.650 CLS Macro-F1), beating every neural encoder — reinforcing that the readable
signal is shallow/orthographic.

---

## Confound control — TF-IDF name-masking ("is dating just the king's name?")

**Data:** balanced MC, 200 draws (same draws as Phase E1). **Targets:** year (Ridge, GroupKFold
by ruler → Spearman) and ruler (logistic → Macro-F1). **Method:** char n-gram TF-IDF (char_wb
2–5), with vs without masking the Akkadian personal-name determinative (`m-…` tokens → `[PN]`),
using the canonical masker `v_1/src/linear_probing/name_masking.py`.
**Source:** `tfidf_namemask_results.json` — these are the LATEST numbers and **supersede** any
older 0.391 figure in the report's E1d cross-reference text.

| Cleaning | Condition | Year Spearman ± std | Year MAE | Ruler Macro-F1 ± std |
|---|---|---|---|---|
| tier0 | unmasked | 0.3545 ± 0.0686 | 43.92 | 0.6496 ± 0.0368 |
| tier0 | **masked** | **0.4002 ± 0.0622** | 42.92 | **0.5273 ± 0.0406** |
| maximal | unmasked | 0.2660 ± 0.0783 | 47.53 | 0.4980 ± 0.0403 |
| maximal | **masked** | **0.2676 ± 0.0863** | 47.27 | **0.4633 ± 0.0393** |

**Findings (from the numbers):**
1. **Masking names does NOT hurt dating.** tier0 year Spearman *rises* 0.355 → 0.400 (within CI);
   maximal essentially unchanged 0.266 → 0.268.
2. **Masking names DOES cost ruler-ID.** tier0 ruler Macro-F1 drops 0.650 → 0.527 (−0.123);
   maximal drops 0.498 → 0.463 (−0.035).
3. So a **name-masked bag of character n-grams still dates as well as a 32B LLM or a domain
   encoder** — the dating signal is orthographic/spelling drift, not the explicit king's name.

**Caveat (report):** masking removes only *explicitly determined* personal names; theophoric /
logographic name elements that double as period vocabulary (e.g. `na-bi`=Nabû) are not masked
and may legitimately still carry chronological information.

**Note for the advisor — discrepancy resolved:** the report's E1d text cross-references TF-IDF
tier0 masked year Spearman as "0.391"; the latest JSON value is **0.400 ± 0.062**. The unmasked
tier0 value (0.355 ± 0.069) matches between report and JSON. Use the JSON figures (0.355 → 0.400;
ruler 0.650 → 0.527) as authoritative.

---

## Overall narrative (the thesis in one paragraph)

Dating Akkadian cuneiform is, to first order, a **shallow orthographic-drift** task: a
character-n-gram TF-IDF baseline — with personal names explicitly masked out — dates fragments
as accurately (balanced year Spearman ≈ 0.40) as a frontier-scale 32B multilingual LLM
(≈ 0.40) or a domain-finetuned Akkadian encoder (≈ 0.41), and these are statistically
indistinguishable under class balancing. The signal is therefore neither name-memorization
(it survives name masking) nor a product of neural scale (bigger Qwen3 does not win — on
balanced Ridge it comes *last*, beaten by a tiny Akkadian MLM and by TF-IDF). The apparent
scale advantage seen on the raw imbalanced corpus (qwen3_32b 0.511 > Thalesian 0.467) is an
imbalance artifact that vanishes once draws are ruler-balanced. The genuinely interesting
neural result is **geometric, not predictive**: the chronology is encoded as a coherent
1-dimensional *temporal manifold* in representation space — most cleanly in the *untrained*
Qwen2.5-7B's first (token-embedding) layer (pacc 0.731, LORO drop 0.008 STRONG, century
centroids ordered along a smooth spline with arc-length Spearman 1.0), and also, less sharply,
in Thalesian's mid layers. So the contribution of Round 3 is to reframe the result away from
"which model dates best" (answer: none, it's shallow) and toward "centuries are laid out as a
smooth curved timeline inside the model, readable without any supervision."

---

## Audit notes / discrepancies found

1. **Name-masking figure mismatch (resolved above):** report E1d says masked tier0 year
   Spearman = 0.391; the authoritative JSON says **0.400 ± 0.062**. Ruler Macro-F1 0.650 → 0.527
   matches the task brief and the JSON.
2. **Balanced-MC year-regression numbers (E1c/E1d) have no standalone scoreboard JSON** in the
   audited directories. `phase0_summary.json` holds only the balanced *ruler-classification*
   Macro-F1 values. The PLS/Ridge balanced-year Spearman figures (0.411, 0.399, 0.408, 0.326,
   etc.) exist only in the prose report and are labelled REPORT-only here. Recommend exporting a
   `e1_balanced_year_scoreboard.json` for reproducibility.
3. **Full-set qwen3_* PLS/Ridge best-layer figures** likewise live only in the report; the raw
   per-layer files (`pls_results_qwen3_*.json`, `cls_numeric_results_qwen3_*.json`, ~5MB each)
   exist but were not distilled into a best-layer scoreboard. Verified the Thalesian / qwen /
   akk300m full-set PLS year-raw numbers directly against `pls_best_layers.json` (all match).
4. **Phase 0 overall verdict = INDETERMINATE** in the JSON, solely because the random-Qwen
   control was never MC-probed (all probed methods individually PASS their gates). Worth noting
   so the advisor doesn't read "INDETERMINATE" as a failed experiment.
5. **Phase D archive coloring is degenerate** (single-valued colorbar) — the archive metadata
   column appears unusable for visualization; only year/ruler/geodesic colorings are informative.
6. All file paths, fragment counts (1,202 total / 1,193 labeled), and the 8×21 balanced draw
   scheme are internally consistent across `phase_0_inventory.json`, the report, and the JSONs.
