# Nonlinear-recovery check — emin2_qwen3_8b_t0_akk_erase_provenance — erased concept `provenance` (15 classes ≥10 docs, chance 0.07)

| features | probe | balanced acc (mean ± sd over folds) |
|---|---|---|
| X raw | linear | 0.414 ± 0.090 |
| X raw | mlp | 0.405 ± 0.077 |
| X erased | linear | 0.066 ± 0.030 |
| X erased | mlp | 0.417 ± 0.053 |
| head hidden h(X erased) | linear | 0.464 ± 0.060 |
| head hidden h(X erased) | mlp | 0.489 ± 0.059 |
