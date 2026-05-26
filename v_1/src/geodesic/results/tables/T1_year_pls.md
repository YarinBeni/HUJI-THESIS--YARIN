# Test 1 — Year regression, PLS

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
