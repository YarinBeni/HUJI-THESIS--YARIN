# S1 representation probes — ssl::ssl_jepa_cunei400m_both-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.391 ± 0.009 | 0.380 ± 0.006 | 0.200 |
| genre_raw | 71 | 27,187 | 0.019 ± 0.001 | 0.023 ± 0.002 | 0.014 |
| provenance | 16 | 30,419 | 0.063 ± 0.000 | 0.078 ± 0.002 | 0.062 |
| source | 5 | 30,720 | 0.200 ± 0.000 | 0.264 ± 0.007 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.725 ± 0.031 | 0.500 |
| seal | 3 | 328 | 0.333 ± 0.000 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.399 | 0.333 |
| orcc (dated) | 3 | 953 | 0.292 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.158 · permutation null -0.069 ± 0.041 · p = 0.970
k-NN (k=10) period purity 0.409 · chance ≈ 0.341
silhouette (UMAP-2d) -0.101 · null -0.038 ± 0.015 · p = 1.000
