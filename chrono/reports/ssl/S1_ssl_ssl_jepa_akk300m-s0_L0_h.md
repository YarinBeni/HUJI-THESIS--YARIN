# S1 representation probes — ssl::ssl_jepa_akk300m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.879 ± 0.022 | 0.872 ± 0.038 | 0.200 |
| genre_raw | 71 | 27,187 | 0.203 ± 0.006 | 0.190 ± 0.014 | 0.014 |
| provenance | 16 | 30,419 | 0.515 ± 0.025 | 0.470 ± 0.032 | 0.062 |
| source | 5 | 30,720 | 0.943 ± 0.006 | 0.944 ± 0.011 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.988 ± 0.013 | 0.500 |
| seal | 3 | 328 | 0.810 ± 0.088 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.643 | 0.333 |
| orcc (dated) | 3 | 953 | 0.148 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.308 · permutation null -0.030 ± 0.014 · p = 0.000
k-NN (k=10) period purity 0.980 · chance ≈ 0.341
silhouette (UMAP-2d) +0.605 · null -0.055 ± 0.028 · p = 0.000
