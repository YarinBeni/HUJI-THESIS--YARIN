# S1 representation probes — ssl_e2e::e2e_barlow_L-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.878 ± 0.033 | 0.851 ± 0.041 | 0.200 |
| genre_raw | 71 | 27,187 | 0.265 ± 0.002 | 0.231 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.672 ± 0.033 | 0.597 ± 0.029 | 0.062 |
| source | 5 | 30,720 | 0.972 ± 0.008 | 0.969 ± 0.010 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.987 ± 0.003 | 0.500 |
| seal | 3 | 328 | 0.828 ± 0.027 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.360 | 0.333 |
| orcc (dated) | 3 | 953 | 0.123 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.233 · permutation null -0.018 ± 0.007 · p = 0.000
k-NN (k=10) period purity 0.983 · chance ≈ 0.341
silhouette (UMAP-2d) +0.157 · null -0.041 ± 0.019 · p = 0.000
