# S1 representation probes — ssl::ssl_infonce_cunei400m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.891 ± 0.031 | 0.870 ± 0.018 | 0.200 |
| genre_raw | 71 | 27,187 | 0.247 ± 0.006 | 0.230 ± 0.017 | 0.014 |
| provenance | 16 | 30,419 | 0.637 ± 0.005 | 0.624 ± 0.008 | 0.062 |
| source | 5 | 30,720 | 0.960 ± 0.006 | 0.960 ± 0.005 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.993 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.767 ± 0.046 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.634 | 0.333 |
| orcc (dated) | 3 | 953 | 0.101 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.025 · permutation null -0.015 ± 0.011 · p = 0.000
k-NN (k=10) period purity 0.976 · chance ≈ 0.341
silhouette (UMAP-2d) +0.619 · null -0.057 ± 0.030 · p = 0.000
