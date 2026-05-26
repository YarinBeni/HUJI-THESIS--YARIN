# Test 3 — Ruler classification

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
