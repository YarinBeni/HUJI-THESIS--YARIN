# S1 representation probes — ssl_hyb::hyb_barlow_M_thalesian_cunei400m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.799 ± 0.021 | 0.780 ± 0.024 | 0.200 |
| genre_raw | 71 | 27,187 | 0.128 ± 0.004 | 0.136 ± 0.004 | 0.014 |
| provenance | 16 | 30,419 | 0.432 ± 0.024 | 0.454 ± 0.025 | 0.062 |
| source | 5 | 30,720 | 0.917 ± 0.009 | 0.920 ± 0.010 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.957 ± 0.017 | 0.500 |
| seal | 3 | 328 | 0.708 ± 0.011 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.610 | 0.333 |
| orcc (dated) | 3 | 953 | 0.167 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.203 · permutation null -0.034 ± 0.016 · p = 0.000
k-NN (k=10) period purity 0.960 · chance ≈ 0.341
silhouette (UMAP-2d) +0.399 · null -0.054 ± 0.029 · p = 0.000
