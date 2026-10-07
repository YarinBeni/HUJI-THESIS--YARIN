# S1 representation probes — ssl::ssl_barlow_cunei400m_wdated-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.905 ± 0.065 | 0.885 ± 0.033 | 0.200 |
| genre_raw | 71 | 27,187 | 0.261 ± 0.009 | 0.218 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.671 ± 0.013 | 0.648 ± 0.014 | 0.062 |
| source | 5 | 30,720 | 0.968 ± 0.005 | 0.970 ± 0.006 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.994 ± 0.007 | 0.500 |
| seal | 3 | 328 | 0.810 ± 0.032 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.657 | 0.333 |
| orcc (dated) | 3 | 953 | 0.257 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.111 · permutation null -0.020 ± 0.014 · p = 0.000
k-NN (k=10) period purity 0.984 · chance ≈ 0.341
silhouette (UMAP-2d) +0.680 · null -0.055 ± 0.029 · p = 0.000
