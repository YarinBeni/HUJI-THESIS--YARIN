# S1 representation probes — ssl_e2e::e2e_jepa_XL-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.855 ± 0.025 | 0.822 ± 0.021 | 0.200 |
| genre_raw | 71 | 27,187 | 0.265 ± 0.005 | 0.223 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.618 ± 0.026 | 0.597 ± 0.019 | 0.062 |
| source | 5 | 30,720 | 0.946 ± 0.006 | 0.936 ± 0.006 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.988 ± 0.010 | 0.500 |
| seal | 3 | 328 | 0.677 ± 0.055 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.514 | 0.333 |
| orcc (dated) | 3 | 953 | 0.106 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.068 · permutation null -0.011 ± 0.007 · p = 0.000
k-NN (k=10) period purity 0.949 · chance ≈ 0.341
silhouette (UMAP-2d) +0.110 · null -0.039 ± 0.019 · p = 0.000
