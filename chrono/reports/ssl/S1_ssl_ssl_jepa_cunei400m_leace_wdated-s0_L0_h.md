# S1 representation probes — ssl::ssl_jepa_cunei400m_leace_wdated-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.573 ± 0.004 | 0.579 ± 0.022 | 0.200 |
| genre_raw | 71 | 27,187 | 0.068 ± 0.001 | 0.085 ± 0.003 | 0.014 |
| provenance | 16 | 30,419 | 0.065 ± 0.000 | 0.203 ± 0.027 | 0.062 |
| source | 5 | 30,720 | 0.207 ± 0.001 | 0.440 ± 0.071 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.930 ± 0.014 | 0.500 |
| seal | 3 | 328 | 0.573 ± 0.026 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.387 | 0.333 |
| orcc (dated) | 3 | 953 | 0.058 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.122 · permutation null -0.061 ± 0.038 · p = 0.910
k-NN (k=10) period purity 0.428 · chance ≈ 0.341
silhouette (UMAP-2d) -0.096 · null -0.044 ± 0.019 · p = 0.980
