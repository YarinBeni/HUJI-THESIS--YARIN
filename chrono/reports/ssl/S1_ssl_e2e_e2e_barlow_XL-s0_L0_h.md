# S1 representation probes — ssl_e2e::e2e_barlow_XL-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.856 ± 0.029 | 0.852 ± 0.032 | 0.200 |
| genre_raw | 71 | 27,187 | 0.260 ± 0.006 | 0.208 ± 0.021 | 0.014 |
| provenance | 16 | 30,419 | 0.659 ± 0.020 | 0.609 ± 0.028 | 0.062 |
| source | 5 | 30,720 | 0.972 ± 0.003 | 0.968 ± 0.003 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.987 ± 0.009 | 0.500 |
| seal | 3 | 328 | 0.782 ± 0.047 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.378 | 0.333 |
| orcc (dated) | 3 | 953 | 0.282 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.163 · permutation null -0.014 ± 0.004 · p = 0.000
k-NN (k=10) period purity 0.982 · chance ≈ 0.341
silhouette (UMAP-2d) +0.067 · null -0.039 ± 0.018 · p = 0.000
