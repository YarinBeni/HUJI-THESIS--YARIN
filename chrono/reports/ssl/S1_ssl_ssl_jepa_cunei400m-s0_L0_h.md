# S1 representation probes — ssl::ssl_jepa_cunei400m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.809 ± 0.019 | 0.766 ± 0.011 | 0.200 |
| genre_raw | 71 | 27,187 | 0.136 ± 0.002 | 0.139 ± 0.004 | 0.014 |
| provenance | 16 | 30,419 | 0.447 ± 0.019 | 0.324 ± 0.030 | 0.062 |
| source | 5 | 30,720 | 0.924 ± 0.011 | 0.899 ± 0.028 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.964 ± 0.020 | 0.500 |
| seal | 3 | 328 | 0.700 ± 0.041 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.661 | 0.333 |
| orcc (dated) | 3 | 953 | 0.146 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.175 · permutation null -0.054 ± 0.032 · p = 0.000
k-NN (k=10) period purity 0.890 · chance ≈ 0.341
silhouette (UMAP-2d) +0.207 · null -0.047 ± 0.026 · p = 0.000
