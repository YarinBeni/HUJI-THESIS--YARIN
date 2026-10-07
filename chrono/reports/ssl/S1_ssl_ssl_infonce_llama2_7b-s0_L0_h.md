# S1 representation probes — ssl::ssl_infonce_llama2_7b-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.852 ± 0.016 | 0.829 ± 0.037 | 0.200 |
| genre_raw | 71 | 27,187 | 0.246 ± 0.004 | 0.216 ± 0.013 | 0.014 |
| provenance | 16 | 30,419 | 0.620 ± 0.035 | 0.588 ± 0.023 | 0.062 |
| source | 5 | 30,720 | 0.973 ± 0.007 | 0.976 ± 0.004 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.993 ± 0.009 | 0.500 |
| seal | 3 | 328 | 0.753 ± 0.032 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.422 | 0.333 |
| orcc (dated) | 3 | 953 | 0.120 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.056 · permutation null -0.023 ± 0.018 · p = 0.925
k-NN (k=10) period purity 0.973 · chance ≈ 0.341
silhouette (UMAP-2d) +0.621 · null -0.063 ± 0.034 · p = 0.000
