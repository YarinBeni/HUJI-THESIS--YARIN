# S1 representation probes — ssl_hyb::hyb_jepa_S_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.862 ± 0.054 | 0.859 ± 0.041 | 0.200 |
| genre_raw | 71 | 27,187 | 0.239 ± 0.003 | 0.203 ± 0.016 | 0.014 |
| provenance | 16 | 30,419 | 0.643 ± 0.017 | 0.587 ± 0.017 | 0.062 |
| source | 5 | 30,720 | 0.979 ± 0.004 | 0.978 ± 0.007 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.992 ± 0.008 | 0.500 |
| seal | 3 | 328 | 0.702 ± 0.041 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.417 | 0.333 |
| orcc (dated) | 3 | 953 | 0.048 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.102 · permutation null -0.014 ± 0.005 · p = 0.000
k-NN (k=10) period purity 0.979 · chance ≈ 0.341
silhouette (UMAP-2d) +0.611 · null -0.048 ± 0.028 · p = 0.000
