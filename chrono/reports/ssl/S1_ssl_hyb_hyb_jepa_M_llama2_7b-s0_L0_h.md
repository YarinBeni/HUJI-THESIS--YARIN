# S1 representation probes — ssl_hyb::hyb_jepa_M_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.860 ± 0.027 | 0.842 ± 0.021 | 0.200 |
| genre_raw | 71 | 27,187 | 0.226 ± 0.007 | 0.210 ± 0.010 | 0.014 |
| provenance | 16 | 30,419 | 0.650 ± 0.031 | 0.595 ± 0.016 | 0.062 |
| source | 5 | 30,720 | 0.984 ± 0.004 | 0.983 ± 0.004 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.996 ± 0.005 | 0.500 |
| seal | 3 | 328 | 0.759 ± 0.044 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.595 | 0.333 |
| orcc (dated) | 3 | 953 | 0.052 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.115 · permutation null -0.015 ± 0.006 · p = 0.000
k-NN (k=10) period purity 0.983 · chance ≈ 0.341
silhouette (UMAP-2d) +0.496 · null -0.044 ± 0.023 · p = 0.000
