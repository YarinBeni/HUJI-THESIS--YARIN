# S1 representation probes — ssl::ssl_jepa_cunei400m_wdated-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.827 ± 0.021 | 0.807 ± 0.042 | 0.200 |
| genre_raw | 71 | 27,187 | 0.127 ± 0.001 | 0.132 ± 0.011 | 0.014 |
| provenance | 16 | 30,419 | 0.413 ± 0.025 | 0.338 ± 0.014 | 0.062 |
| source | 5 | 30,720 | 0.915 ± 0.009 | 0.856 ± 0.023 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.967 ± 0.014 | 0.500 |
| seal | 3 | 328 | 0.757 ± 0.045 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.739 | 0.333 |
| orcc (dated) | 3 | 953 | 0.118 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.215 · permutation null -0.054 ± 0.032 · p = 0.000
k-NN (k=10) period purity 0.893 · chance ≈ 0.341
silhouette (UMAP-2d) +0.274 · null -0.047 ± 0.022 · p = 0.000
