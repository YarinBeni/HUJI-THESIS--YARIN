# S1 representation probes — ssl::ssl_barlow_cunei400m_leace-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.631 ± 0.024 | 0.790 ± 0.017 | 0.200 |
| genre_raw | 71 | 27,187 | 0.217 ± 0.004 | 0.186 ± 0.007 | 0.014 |
| provenance | 16 | 30,419 | 0.341 ± 0.014 | 0.589 ± 0.027 | 0.062 |
| source | 5 | 30,720 | 0.213 ± 0.002 | 0.875 ± 0.016 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.994 ± 0.007 | 0.500 |
| seal | 3 | 328 | 0.767 ± 0.040 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.264 | 0.333 |
| orcc (dated) | 3 | 953 | 0.061 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.054 · permutation null -0.022 ± 0.017 · p = 0.915
k-NN (k=10) period purity 0.894 · chance ≈ 0.341
silhouette (UMAP-2d) +0.303 · null -0.054 ± 0.028 · p = 0.000
