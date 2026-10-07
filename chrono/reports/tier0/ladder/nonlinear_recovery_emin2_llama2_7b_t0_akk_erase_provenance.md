# Nonlinear-recovery check — emin2_llama2_7b_t0_akk_erase_provenance — erased concept `provenance` (15 classes ≥10 docs, chance 0.07)

| features | probe | balanced acc (mean ± sd over folds) |
|---|---|---|
| X raw | linear | 0.423 ± 0.101 |
| X raw | mlp | 0.435 ± 0.069 |
| X erased | linear | 0.052 ± 0.022 |
| X erased | mlp | 0.354 ± 0.063 |
| head hidden h(X erased) | linear | 0.418 ± 0.053 |
| head hidden h(X erased) | mlp | 0.390 ± 0.107 |
