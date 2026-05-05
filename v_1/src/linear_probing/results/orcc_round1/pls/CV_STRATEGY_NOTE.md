# PLS Year-Regression: CV Strategy Note

**Date:** 2026-05-05  
**Pipeline version:** orcc_round1 PLS jobs 5561–5563 + TF-IDF local

---

## Setup

Predicting year-of-composition from hidden-state activations using Partial Least Squares
regression (PLSRegression, sklearn). ORCC corpus: 893 labeled fragments across 38 unique
rulers, 45 unique years. Year is largely a function of ruler (median 1 year per ruler).

**CV strategy chosen:** `GroupKFold(n_splits=5)`, grouped by ruler.  
**Rationale:** Prevents the model from seeing a ruler's fragments in both train and test,
giving an estimate of how well embeddings generalize *across rulers* (not just across
fragments of the same ruler).

---

## Known Issues and Fixes Applied

### Issue 1 — Degenerate folds (methodology)

Because year ≈ ruler and GroupKFold holds out entire rulers, test folds sometimes contain
fragments from a single ruler with a single year → constant `y_test`. Spearman correlation
is undefined for a constant series (returns NaN). The same applies to the shuffled-y
baseline.

**Observed:** 2/5 folds are consistently degenerate across all configs.  
**Effect:** 3 valid folds remain per config.

### Issue 2 — NaN propagation in aggregation (bug, now fixed)

`np.mean([nan, nan, 0.15, -0.47, -0.19])` = NaN. The original aggregator
(`06_aggregate_pls.py`) used the pre-computed `spearman_mean` scalar (which was already
NaN-propagated). The `pls_best_layers.json` previously showed all-NaN Spearman and
best_layer=0 everywhere.

**Fix (2026-05-05):** `06_aggregate_pls.py` now recomputes all metrics from stored
fold-level arrays (`spearman_folds`, `r2_folds`, etc.), excluding degenerate folds
identified by NaN Spearman. `n_valid_folds` is now tracked in both JSON outputs and the
printed table. `pls_utils.py` was also updated to emit correct NaN-safe means for future
runs.

### Issue 3 — Catastrophic R² (data structure)

Even in valid folds, R² is deeply negative (often −100 to −6000). Cause: PLS trained on
rulers A–Z predicting ruler W whose year falls outside the training year range → wild
extrapolation. R² is thus uninformative as a primary metric here. All R² plots are clipped
at −10 for readability.

**Spearman** is the primary metric (rank-based, robust to extrapolation).

### Issue 4 — Shuffled baseline not recoverable

The shuffled-y baseline (`shuffled_spearman_mean`) is stored as a pre-computed scalar and
also suffers NaN propagation from degenerate folds. No fold-level shuffled data was stored
in results v1, so the baseline cannot be recovered without re-running cluster jobs.  
`pls_utils.py` now stores fold-level shuffled data for future runs.

---

## Results Summary (Spearman, 3/5 valid folds)

| Method | Config | Best layer | Best k | Spearman |
|--------|--------|-----------|--------|----------|
| qwen   | tier0, mean, raw | L5 | 5 | +0.121 |
| random | tier0, mean, raw | L12 | 2 | **+0.184** |
| tfidf  | tier0, na, log   | L0 | 1 | +0.181 |
| mlm    | tier0, mean, raw | L2 | 2 | −0.115 |

**Key finding:** Random-weight initialization (Qwen architecture, seed=42) and TF-IDF match
or exceed the pretrained Qwen model. This indicates that the linear PLS subspace does not
encode year-of-composition more strongly than surface n-gram features. The weak Spearman
values (max ≈ 0.18) suggest near-absence of recoverable temporal signal under this CV
scheme.

---

## Alternative CV Strategies (considered, not implemented)

| Option | Pro | Con |
|--------|-----|-----|
| **Random 5-fold** (no grouping) | All 5 folds valid; cleaner metrics | Fragments from same ruler in train+test → inflated signal; not independent |
| **GroupKFold by period** (OB/NA/LB) | Coarser groups, more samples per fold | Fewer folds possible; ORCC period labels may be incomplete |
| **Aggregate by ruler** (38 rows, LOOCV) | Cleanest unit of analysis; matches data structure | Only 38 data points; high variance |
| **Ordinal classification** (45 classes) | Avoids extrapolation problem | Reframes the task; loses regression metrics |

For a direct comparison with El-Shangiti et al. (NAACL 2025), random 5-fold would be the
closest match to their setup. A follow-up run with random CV would clarify how much of the
0.18 Spearman is within-ruler signal vs. across-ruler generalization.

---

## Files

- `pls_best_layers.json` — best layer per (method, cleaning, pooling, year_transform); includes `n_valid_folds`
- `pls_layer_curves.json` — all layer × k Spearman/R²/MAE curves
- `figures/` — 22 per-config PNGs + 2 combined best-of PNGs
