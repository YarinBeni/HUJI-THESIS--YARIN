# S1 representation probes — ssl::ssl_jepa_cunei400m_adv-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.806 ± 0.015 | 0.775 ± 0.045 | 0.200 |
| genre_raw | 71 | 27,187 | 0.107 ± 0.004 | 0.121 ± 0.004 | 0.014 |
| provenance | 16 | 30,419 | 0.380 ± 0.014 | 0.367 ± 0.003 | 0.062 |
| source | 5 | 30,720 | 0.925 ± 0.010 | 0.922 ± 0.012 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.910 ± 0.019 | 0.500 |
| seal | 3 | 328 | 0.732 ± 0.052 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.660 | 0.333 |
| orcc (dated) | 3 | 953 | 0.259 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.035 · permutation null -0.049 ± 0.027 · p = 0.000
k-NN (k=10) period purity 0.882 · chance ≈ 0.341
silhouette (UMAP-2d) +0.116 · null -0.052 ± 0.025 · p = 0.000
