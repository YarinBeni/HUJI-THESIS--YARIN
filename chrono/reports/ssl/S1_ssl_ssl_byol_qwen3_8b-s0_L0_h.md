# S1 representation probes — ssl::ssl_byol_qwen3_8b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.797 ± 0.010 | 0.781 ± 0.023 | 0.200 |
| genre_raw | 71 | 27,187 | 0.157 ± 0.005 | 0.153 ± 0.011 | 0.014 |
| provenance | 16 | 30,419 | 0.469 ± 0.016 | 0.420 ± 0.028 | 0.062 |
| source | 5 | 30,720 | 0.939 ± 0.005 | 0.931 ± 0.012 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.931 ± 0.020 | 0.500 |
| seal | 3 | 328 | 0.682 ± 0.033 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.623 | 0.333 |
| orcc (dated) | 3 | 953 | 0.098 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.239 · permutation null -0.039 ± 0.021 · p = 0.000
k-NN (k=10) period purity 0.939 · chance ≈ 0.341
silhouette (UMAP-2d) +0.378 · null -0.054 ± 0.030 · p = 0.000
