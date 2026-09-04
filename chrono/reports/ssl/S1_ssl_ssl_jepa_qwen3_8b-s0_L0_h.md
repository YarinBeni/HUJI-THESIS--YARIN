# S1 representation probes — ssl::ssl_jepa_qwen3_8b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.809 ± 0.020 | 0.802 ± 0.028 | 0.200 |
| genre_raw | 71 | 27,187 | 0.167 ± 0.008 | 0.155 ± 0.006 | 0.014 |
| provenance | 16 | 30,419 | 0.493 ± 0.010 | 0.413 ± 0.018 | 0.062 |
| source | 5 | 30,720 | 0.949 ± 0.009 | 0.943 ± 0.009 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.954 ± 0.008 | 0.500 |
| seal | 3 | 328 | 0.679 ± 0.034 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.477 | 0.333 |
| orcc (dated) | 3 | 953 | 0.067 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.082 · permutation null -0.037 ± 0.018 · p = 0.000
k-NN (k=10) period purity 0.942 · chance ≈ 0.341
silhouette (UMAP-2d) +0.225 · null -0.046 ± 0.022 · p = 0.000
