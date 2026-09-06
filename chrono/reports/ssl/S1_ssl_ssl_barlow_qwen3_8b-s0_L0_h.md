# S1 representation probes — ssl::ssl_barlow_qwen3_8b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.820 ± 0.014 | 0.801 ± 0.016 | 0.200 |
| genre_raw | 71 | 27,187 | 0.206 ± 0.004 | 0.188 ± 0.015 | 0.014 |
| provenance | 16 | 30,419 | 0.586 ± 0.025 | 0.538 ± 0.024 | 0.062 |
| source | 5 | 30,720 | 0.964 ± 0.005 | 0.962 ± 0.008 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.977 ± 0.008 | 0.500 |
| seal | 3 | 328 | 0.653 ± 0.046 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.597 | 0.333 |
| orcc (dated) | 3 | 953 | 0.096 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.197 · permutation null -0.026 ± 0.016 · p = 0.000
k-NN (k=10) period purity 0.966 · chance ≈ 0.341
silhouette (UMAP-2d) +0.455 · null -0.058 ± 0.033 · p = 0.000
