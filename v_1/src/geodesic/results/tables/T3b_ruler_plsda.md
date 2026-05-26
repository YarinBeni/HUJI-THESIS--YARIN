# Test 3b — Ruler classification, PLS-DA

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
