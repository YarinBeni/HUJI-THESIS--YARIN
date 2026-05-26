# Test 5 — LORO (Leave-One-Ruler-Out)

**What it is:** is the manifold a real *timeline*, or just "each ruler is its own blob near its
date"? Refit the Isomap manifold with **one ruler's fragments removed**, drop those held-out
fragments onto it, re-measure pacc. Small drop = genuine temporal axis.

**Data & split:** held out one ruler at a time (11 rulers); manifold fit on the other 10.
`drop` = pacc_full − mean(pacc over held-out rulers). STRONG if drop < 0.10.

**CSVs:** `T5_loro.csv` (one row per config, summary drop) and `T5_loro_per_ruler.csv`
(per held-out ruler: `pacc_loro` = held-in pacc, `pacc_cross` = held-out fragments' pacc, `n`).
