# S1 representation probes — ssl_hyb::hyb_barlow_M_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.875 ± 0.023 | 0.858 ± 0.036 | 0.200 |
| genre_raw | 71 | 27,187 | 0.247 ± 0.005 | 0.223 ± 0.007 | 0.014 |
| provenance | 16 | 30,419 | 0.617 ± 0.019 | 0.569 ± 0.019 | 0.062 |
| source | 5 | 30,720 | 0.988 ± 0.003 | 0.989 ± 0.004 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.993 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.782 ± 0.029 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.361 | 0.333 |
| orcc (dated) | 3 | 953 | 0.145 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.193 · permutation null -0.015 ± 0.005 · p = 0.000
k-NN (k=10) period purity 0.984 · chance ≈ 0.341
silhouette (UMAP-2d) +0.538 · null -0.047 ± 0.024 · p = 0.000
