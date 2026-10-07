# S1 representation probes — ssl::ssl_jepa_cunei400m_leace-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.576 ± 0.005 | 0.596 ± 0.024 | 0.200 |
| genre_raw | 71 | 27,187 | 0.074 ± 0.001 | 0.094 ± 0.010 | 0.014 |
| provenance | 16 | 30,419 | 0.066 ± 0.001 | 0.235 ± 0.039 | 0.062 |
| source | 5 | 30,720 | 0.207 ± 0.001 | 0.568 ± 0.033 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.938 ± 0.010 | 0.500 |
| seal | 3 | 328 | 0.636 ± 0.045 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.357 | 0.333 |
| orcc (dated) | 3 | 953 | 0.078 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.131 · permutation null -0.060 ± 0.038 · p = 0.935
k-NN (k=10) period purity 0.453 · chance ≈ 0.341
silhouette (UMAP-2d) -0.096 · null -0.046 ± 0.019 · p = 0.990
