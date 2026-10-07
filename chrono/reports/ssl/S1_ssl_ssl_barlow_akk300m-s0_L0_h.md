# S1 representation probes — ssl::ssl_barlow_akk300m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.871 ± 0.020 | 0.865 ± 0.036 | 0.200 |
| genre_raw | 71 | 27,187 | 0.258 ± 0.017 | 0.207 ± 0.016 | 0.014 |
| provenance | 16 | 30,419 | 0.623 ± 0.010 | 0.586 ± 0.039 | 0.062 |
| source | 5 | 30,720 | 0.947 ± 0.009 | 0.953 ± 0.008 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.996 ± 0.005 | 0.500 |
| seal | 3 | 328 | 0.765 ± 0.070 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.662 | 0.333 |
| orcc (dated) | 3 | 953 | 0.205 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.086 · permutation null -0.025 ± 0.018 · p = 0.000
k-NN (k=10) period purity 0.979 · chance ≈ 0.341
silhouette (UMAP-2d) +0.630 · null -0.055 ± 0.028 · p = 0.000
