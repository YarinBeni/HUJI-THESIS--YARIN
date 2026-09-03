# S1 representation probes — Qwen/Qwen3-8B::L27::mean

texts 30,729 · PCA 256 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.856 ± 0.021 | 0.798 ± 0.011 | 0.200 |
| genre_raw | 71 | 27,187 | 0.249 ± 0.009 | 0.199 ± 0.007 | 0.014 |
| provenance | 16 | 30,419 | 0.655 ± 0.011 | 0.657 ± 0.020 | 0.062 |
| source | 5 | 30,720 | 0.963 ± 0.003 | 0.970 ± 0.004 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.990 ± 0.005 | 0.500 |
| seal | 3 | 328 | 0.761 ± 0.043 | 0.333 |

## Period probe, HELD-OUT source (train on the others, linear)

| held out | n test | balanced acc | chance |
|---|---|---|---|

## Geometry (period)

silhouette (raw space) +0.080 · permutation null -0.025 ± 0.016 · p = 0.000
k-NN (k=10) period purity 0.964 · chance ≈ 0.341
silhouette (UMAP-2d) +0.462 · null -0.063 ± 0.034 · p = 0.000
