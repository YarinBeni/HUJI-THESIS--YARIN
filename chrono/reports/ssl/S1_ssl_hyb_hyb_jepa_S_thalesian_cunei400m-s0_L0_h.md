# S1 representation probes — ssl_hyb::hyb_jepa_S_thalesian_cunei400m-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.836 ± 0.031 | 0.826 ± 0.023 | 0.200 |
| genre_raw | 71 | 27,187 | 0.228 ± 0.004 | 0.192 ± 0.007 | 0.014 |
| provenance | 16 | 30,419 | 0.521 ± 0.028 | 0.520 ± 0.023 | 0.062 |
| source | 5 | 30,720 | 0.935 ± 0.006 | 0.939 ± 0.009 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.989 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.694 ± 0.064 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.501 | 0.333 |
| orcc (dated) | 3 | 953 | 0.100 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.045 · permutation null -0.007 ± 0.003 · p = 0.000
k-NN (k=10) period purity 0.945 · chance ≈ 0.341
silhouette (UMAP-2d) +0.274 · null -0.041 ± 0.022 · p = 0.000
