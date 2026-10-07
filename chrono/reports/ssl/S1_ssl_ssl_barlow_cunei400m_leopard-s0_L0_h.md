# S1 representation probes — ssl::ssl_barlow_cunei400m_leopard-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.893 ± 0.047 | 0.862 ± 0.042 | 0.200 |
| genre_raw | 71 | 27,187 | 0.266 ± 0.007 | 0.221 ± 0.017 | 0.014 |
| provenance | 16 | 30,419 | 0.667 ± 0.011 | 0.623 ± 0.022 | 0.062 |
| source | 5 | 30,720 | 0.962 ± 0.006 | 0.952 ± 0.005 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.992 ± 0.007 | 0.500 |
| seal | 3 | 328 | 0.752 ± 0.021 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.678 | 0.333 |
| orcc (dated) | 3 | 953 | 0.191 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.006 · permutation null -0.020 ± 0.015 · p = 0.000
k-NN (k=10) period purity 0.921 · chance ≈ 0.341
silhouette (UMAP-2d) +0.291 · null -0.053 ± 0.026 · p = 0.000
