# S1 representation probes — Qwen/Qwen3-8B::L18::mean

texts 30,729 · PCA 256 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.838 ± 0.017 | 0.813 ± 0.022 | 0.200 |
| genre_raw | 71 | 27,187 | 0.233 ± 0.010 | 0.189 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.631 ± 0.031 | 0.619 ± 0.018 | 0.062 |
| source | 5 | 30,720 | 0.963 ± 0.005 | 0.965 ± 0.004 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.988 ± 0.008 | 0.500 |
| seal | 3 | 328 | 0.702 ± 0.057 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.415 | 0.333 |
| orcc (dated) | 3 | 953 | 0.114 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.052 · permutation null -0.024 ± 0.016 · p = 0.000
k-NN (k=10) period purity 0.957 · chance ≈ 0.341
silhouette (UMAP-2d) +0.376 · null -0.061 ± 0.031 · p = 0.000
