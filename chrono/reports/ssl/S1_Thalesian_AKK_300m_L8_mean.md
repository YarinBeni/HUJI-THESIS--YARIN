# S1 representation probes — Thalesian/AKK_300m::L8::mean

texts 30,729 · PCA 256 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.865 ± 0.012 | 0.816 ± 0.031 | 0.200 |
| genre_raw | 71 | 27,187 | 0.261 ± 0.009 | 0.211 ± 0.012 | 0.014 |
| provenance | 16 | 30,419 | 0.650 ± 0.022 | 0.634 ± 0.014 | 0.062 |
| source | 5 | 30,720 | 0.939 ± 0.004 | 0.949 ± 0.006 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.991 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.793 ± 0.056 | 0.333 |

## Period probe, HELD-OUT source (train on the others, linear)

| held out | n test | balanced acc | chance |
|---|---|---|---|

## Geometry (period)

silhouette (raw space) +0.113 · permutation null -0.023 ± 0.013 · p = 0.000
k-NN (k=10) period purity 0.969 · chance ≈ 0.341
silhouette (UMAP-2d) +0.625 · null -0.057 ± 0.033 · p = 0.000
