# S1 representation probes — ssl_e2e::e2e_jepa_L-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.870 ± 0.037 | 0.839 ± 0.030 | 0.200 |
| genre_raw | 71 | 27,187 | 0.278 ± 0.007 | 0.245 ± 0.019 | 0.014 |
| provenance | 16 | 30,419 | 0.664 ± 0.018 | 0.632 ± 0.023 | 0.062 |
| source | 5 | 30,720 | 0.951 ± 0.006 | 0.950 ± 0.008 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.991 ± 0.007 | 0.500 |
| seal | 3 | 328 | 0.765 ± 0.060 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.603 | 0.333 |
| orcc (dated) | 3 | 953 | 0.104 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.060 · permutation null -0.013 ± 0.010 · p = 0.000
k-NN (k=10) period purity 0.959 · chance ≈ 0.341
silhouette (UMAP-2d) +0.013 · null -0.043 ± 0.025 · p = 0.000
