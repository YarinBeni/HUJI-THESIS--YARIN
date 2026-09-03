# S1 representation probes — Thalesian/cuneiformBase-400m::L6::mean

texts 30,729 · PCA 256 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.901 ± 0.035 | 0.843 ± 0.036 | 0.200 |
| genre_raw | 71 | 27,187 | 0.278 ± 0.008 | 0.229 ± 0.015 | 0.014 |
| provenance | 16 | 30,419 | 0.689 ± 0.025 | 0.666 ± 0.022 | 0.062 |
| source | 5 | 30,720 | 0.967 ± 0.005 | 0.970 ± 0.005 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.995 ± 0.007 | 0.500 |
| seal | 3 | 328 | 0.768 ± 0.045 | 0.333 |

## Period probe, HELD-OUT source (train on the others, linear)

| held out | n test | balanced acc | chance |
|---|---|---|---|

## Geometry (period)

silhouette (raw space) +0.138 · permutation null -0.021 ± 0.012 · p = 0.000
k-NN (k=10) period purity 0.983 · chance ≈ 0.341
silhouette (UMAP-2d) +0.645 · null -0.058 ± 0.032 · p = 0.000
