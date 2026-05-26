# Test 4 — Geodesic / Isomap manifold (unsupervised)

**What it is:** rather than *training* a probe, ask whether fragments already lie along a curved
1-D "timeline" in activation space. **Isomap** builds a k-nearest-neighbor graph on the vectors
and "unrolls" it into one coordinate; years are never shown. `ebin` = an alternative
earliest-bin geodesic readout.

**Data & split:** unsupervised (no labels used to fit); evaluated on **all** fragments.
Metrics: **pairwise_acc (pacc)** = of fragment pairs >100yr apart, the fraction the 1-D
coordinate orders correctly (0.5 chance, 1.0 perfect) — the headline; **spearman** of the
coordinate vs year; **neighbor_purity** = fraction of each point's 10 nearest neighbors within
±100yr, with **neighbor_sigma** = σ above a shuffled-label null.

**CSV `T4_geodesic.csv`** — one row per (method, cleaning, pool, layer): all 728 swept configs.
Filter to max `isomap_pairwise_acc` per method for the best-layer leaderboard.
