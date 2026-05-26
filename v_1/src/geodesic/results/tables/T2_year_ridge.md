# Test 2 — Year regression, Ridge

**What it is:** plain L2-penalized linear regression predicting year directly from the
activation vector — a single-direction readout, simpler than PLS.

**Data & split:** same as Test 1 (mean-pooled activations, 5-fold GroupKFold by ruler).
`imbalanced` Ridge was only run for the qwen3_* models; `balanced` exists for every model that
has a `*_cls_numeric` MC summary (mlm, tfidf, qwen3_*).

**CSV `T2_year_ridge.csv`** — one row per (regime, model, cleaning, pool, layer, year_transform).
Metrics: Spearman, R2, MAE. Headline = `regime=balanced, year_transform=raw`.
