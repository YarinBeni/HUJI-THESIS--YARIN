# S1 representation probes — ssl::ssl_barlow_cunei400m_both-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.239 ± 0.005 | 0.209 ± 0.003 | 0.200 |
| genre_raw | 71 | 27,187 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.014 |
| provenance | 16 | 30,419 | 0.062 ± 0.000 | 0.063 ± 0.000 | 0.062 |
| source | 5 | 30,720 | 0.200 ± 0.000 | 0.201 ± 0.001 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.514 ± 0.016 | 0.500 |
| seal | 3 | 328 | 0.333 ± 0.000 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.351 | 0.333 |
| orcc (dated) | 3 | 953 | 0.268 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.167 · permutation null -0.069 ± 0.041 · p = 0.985
k-NN (k=10) period purity 0.379 · chance ≈ 0.341
silhouette (UMAP-2d) -0.100 · null -0.038 ± 0.016 · p = 1.000
