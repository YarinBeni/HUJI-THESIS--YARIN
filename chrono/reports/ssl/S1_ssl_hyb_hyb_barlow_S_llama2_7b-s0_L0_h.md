# S1 representation probes — ssl_hyb::hyb_barlow_S_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.886 ± 0.042 | 0.886 ± 0.035 | 0.200 |
| genre_raw | 71 | 27,187 | 0.251 ± 0.008 | 0.215 ± 0.019 | 0.014 |
| provenance | 16 | 30,419 | 0.658 ± 0.029 | 0.592 ± 0.031 | 0.062 |
| source | 5 | 30,720 | 0.987 ± 0.001 | 0.987 ± 0.001 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.991 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.815 ± 0.064 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.423 | 0.333 |
| orcc (dated) | 3 | 953 | 0.092 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.178 · permutation null -0.014 ± 0.005 · p = 0.000
k-NN (k=10) period purity 0.985 · chance ≈ 0.341
silhouette (UMAP-2d) +0.502 · null -0.044 ± 0.022 · p = 0.000
