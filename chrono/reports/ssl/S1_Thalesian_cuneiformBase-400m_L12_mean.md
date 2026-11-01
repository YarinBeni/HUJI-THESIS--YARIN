# S1 representation probes — Thalesian/cuneiformBase-400m::L12::mean

texts 30,729 · PCA 256 · classes need ≥ 30 docs

## Probes (balanced accuracy, 5-fold, tablet level)

| label | classes | n | linear | MLP | chance |
|---|---|---|---|---|---|
| period_norm | 5 | 6,300 | 0.890 ± 0.041 | 0.823 ± 0.029 | 0.200 |
| genre_raw | 71 | 27,187 | 0.273 ± 0.012 | 0.215 ± 0.012 | 0.014 |
| provenance | 16 | 30,419 | 0.674 ± 0.020 | 0.670 ± 0.023 | 0.062 |
| source | 5 | 30,720 | 0.963 ± 0.009 | 0.963 ± 0.007 | 0.200 |

## Period probe WITHIN source (linear)

| source | classes | n | balanced acc | chance |
|---|---|---|---|---|
| oracc | 2 | 3,414 | 0.993 ± 0.009 | 0.500 |
| seal | 3 | 328 | 0.775 ± 0.054 | 0.333 |

## Period probe, HELD-OUT source (train on the other non-dated sources, linear)

| held out | classes | n test | balanced acc | chance |
|---|---|---|---|---|
| seal | 3 | 292 | 0.668 | 0.333 |
| orcc (dated) | 3 | 953 | 0.179 | 0.333 |

## Geometry (period)

silhouette (raw space) +0.092 · permutation null -0.018 ± 0.011 · p = 0.000
k-NN (k=10) period purity 0.981 · chance ≈ 0.341
silhouette (UMAP-2d) +0.631 · null -0.057 ± 0.029 · p = 0.000
