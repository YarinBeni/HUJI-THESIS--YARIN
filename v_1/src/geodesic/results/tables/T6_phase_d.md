# Test 6 — Phase D visualization (centroid + spline)

**What it is:** a figure/quant check — bin fragments into 100-year windows, take each window's
3-D PCA centroid, fit a smooth spline through the centroids, and measure whether
distance-along-the-curve tracks century order.

**Data & split:** all fragments; 7 populated century bins. Metric = **arc_length_spearman**
(1.0 = the curve threads centuries in perfect order). `bin_centers`/`bin_counts` are `|`-joined.

**CSV `T6_phase_d.csv`** — one row per visualized config. 12 PNGs (4 colorings x 3 configs) live
in `../phase_d/` and are embedded in `../EXPERIMENTS_SUMMARY.md`.
