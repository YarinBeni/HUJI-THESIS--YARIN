# Test 3 — Ruler classification

**What it is:** predict *which ruler* a fragment belongs to (multi-class). The "identify the
king" task — explicit names live here.

**Data & split:** 5-fold **StratifiedKFold** (same rulers in train & test). Metric = Macro-F1
(per-ruler F1 averaged equally; rare rulers count as much as common ones).

**Not apples-to-apples:** `r1_imbalanced` uses 11–41 rulers (chance tiny); `balanced_mc` uses
8 rulers (chance 0.125), so balanced Macro-F1 is mechanically higher. Use the columns to rank
*methods*, not to claim balancing "helped". CSV `T3_ruler_classification.csv` — one row per
(model, cleaning, pool) with both regimes side by side.
