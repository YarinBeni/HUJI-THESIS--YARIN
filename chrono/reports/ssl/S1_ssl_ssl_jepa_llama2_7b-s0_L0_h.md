# S1 representation probes — ssl::ssl_jepa_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.848 ± 0.020 | 0.838 ± 0.025 | 0.200 |
| genre_raw | 71 | 27,187 | 0.201 ± 0.002 | 0.189 ± 0.006 | 0.014 |
| provenance | 16 | 30,419 | 0.544 ± 0.010 | 0.504 ± 0.022 | 0.062 |
| source | 5 | 30,720 | 0.969 ± 0.002 | 0.965 ± 0.005 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.981 ± 0.010 | 0.500 |
| seal | 3 | 328 | 0.712 ± 0.036 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.573 | 0.333 |
| orcc (dated) | 3 | 953 | 0.105 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.277 · permutation null -0.030 ± 0.016 · p = 0.000
k-NN (k=10) period purity 0.969 · chance ≈ 0.341
silhouette (UMAP-2d) +0.454 · null -0.054 ± 0.030 · p = 0.000
