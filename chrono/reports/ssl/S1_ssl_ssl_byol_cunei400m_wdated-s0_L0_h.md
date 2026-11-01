# S1 representation probes — ssl::ssl_byol_cunei400m_wdated-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.826 ± 0.022 | 0.819 ± 0.016 | 0.200 |
| genre_raw | 71 | 27,187 | 0.128 ± 0.003 | 0.137 ± 0.010 | 0.014 |
| provenance | 16 | 30,419 | 0.416 ± 0.009 | 0.325 ± 0.014 | 0.062 |
| source | 5 | 30,720 | 0.926 ± 0.008 | 0.876 ± 0.038 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.962 ± 0.022 | 0.500 |
| seal | 3 | 328 | 0.774 ± 0.073 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.680 | 0.333 |
| orcc (dated) | 3 | 953 | 0.109 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.229 · permutation null -0.056 ± 0.032 · p = 0.000
k-NN (k=10) period purity 0.893 · chance ≈ 0.341
silhouette (UMAP-2d) +0.193 · null -0.046 ± 0.020 · p = 0.000
