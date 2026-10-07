# S1 representation probes — ssl::ssl_jepa_cunei400m_leopard-s0::L0::h

texts 30,729 · PCA 0 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.790 ± 0.018 | 0.786 ± 0.021 | 0.200 |
| genre_raw | 71 | 27,187 | 0.186 ± 0.003 | 0.168 ± 0.016 | 0.014 |
| provenance | 16 | 30,419 | 0.478 ± 0.015 | 0.463 ± 0.019 | 0.062 |
| source | 5 | 30,720 | 0.798 ± 0.010 | 0.820 ± 0.016 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.955 ± 0.006 | 0.500 |
| seal | 3 | 328 | 0.680 ± 0.072 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.593 | 0.333 |
| orcc (dated) | 3 | 953 | 0.076 | 0.333 |

## Geometry (period)

silhouette (raw space) -0.025 · permutation null -0.032 ± 0.025 · p = 0.545
k-NN (k=10) period purity 0.767 · chance ≈ 0.341
silhouette (UMAP-2d) -0.048 · null -0.047 ± 0.024 · p = 0.580
