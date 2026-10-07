# S1 representation probes — ssl::ssl_byol_akk300m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.885 ± 0.024 | 0.848 ± 0.041 | 0.200 |
| genre_raw | 71 | 27,187 | 0.188 ± 0.007 | 0.177 ± 0.006 | 0.014 |
| provenance | 16 | 30,419 | 0.492 ± 0.022 | 0.411 ± 0.014 | 0.062 |
| source | 5 | 30,720 | 0.935 ± 0.011 | 0.922 ± 0.010 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.992 ± 0.012 | 0.500 |
| seal | 3 | 328 | 0.791 ± 0.052 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.664 | 0.333 |
| orcc (dated) | 3 | 953 | 0.143 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.302 · permutation null -0.031 ± 0.014 · p = 0.000
k-NN (k=10) period purity 0.974 · chance ≈ 0.341
silhouette (UMAP-2d) +0.585 · null -0.053 ± 0.026 · p = 0.000
