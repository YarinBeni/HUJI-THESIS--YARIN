# S1 representation probes — ssl::ssl_byol_cunei400m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.840 ± 0.030 | 0.815 ± 0.035 | 0.200 |
| genre_raw | 71 | 27,187 | 0.129 ± 0.003 | 0.125 ± 0.002 | 0.014 |
| provenance | 16 | 30,419 | 0.391 ± 0.004 | 0.354 ± 0.027 | 0.062 |
| source | 5 | 30,720 | 0.918 ± 0.010 | 0.887 ± 0.021 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.980 ± 0.012 | 0.500 |
| seal | 3 | 328 | 0.807 ± 0.079 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.643 | 0.333 |
| orcc (dated) | 3 | 953 | 0.159 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.192 · permutation null -0.050 ± 0.027 · p = 0.000
k-NN (k=10) period purity 0.903 · chance ≈ 0.341
silhouette (UMAP-2d) +0.205 · null -0.051 ± 0.025 · p = 0.000
