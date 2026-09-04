# S1 representation probes — ssl::ssl_barlow_cunei400m_adv-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.899 ± 0.030 | 0.842 ± 0.014 | 0.200 |
| genre_raw | 71 | 27,187 | 0.211 ± 0.003 | 0.204 ± 0.010 | 0.014 |
| provenance | 16 | 30,419 | 0.621 ± 0.017 | 0.513 ± 0.041 | 0.062 |
| source | 5 | 30,720 | 0.959 ± 0.004 | 0.960 ± 0.013 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.992 ± 0.010 | 0.500 |
| seal | 3 | 328 | 0.801 ± 0.015 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.635 | 0.333 |
| orcc (dated) | 3 | 953 | 0.165 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.007 · permutation null -0.046 ± 0.025 · p = 0.000
k-NN (k=10) period purity 0.956 · chance ≈ 0.341
silhouette (UMAP-2d) +0.206 · null -0.048 ± 0.027 · p = 0.000
