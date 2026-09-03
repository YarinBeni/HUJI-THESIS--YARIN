# S1 representation probes — ssl::ssl_infonce_qwen3_8b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.852 ± 0.028 | 0.823 ± 0.034 | 0.200 |
| genre_raw | 71 | 27,187 | 0.234 ± 0.008 | 0.217 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.632 ± 0.023 | 0.596 ± 0.016 | 0.062 |
| source | 5 | 30,720 | 0.968 ± 0.006 | 0.974 ± 0.003 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.990 ± 0.009 | 0.500 |
| seal | 3 | 328 | 0.722 ± 0.027 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.451 | 0.333 |
| orcc (dated) | 3 | 953 | 0.094 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.025 · permutation null -0.025 ± 0.019 · p = 0.610
k-NN (k=10) period purity 0.975 · chance ≈ 0.341
silhouette (UMAP-2d) +0.575 · null -0.062 ± 0.034 · p = 0.000
