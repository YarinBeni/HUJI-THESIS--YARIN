# Test 8 — Metadata-only year baseline (B1, the floor)

**What it is:** predicts year from **metadata alone** — one-hot {ruler, provenance,
period} (genre excluded, single-valued) — under the *same* Ridge GroupKFold-by-ruler
protocol as Test 2. Every embedding's year readout must beat this floor to claim it
learned anything beyond find-site + period bookkeeping.

**Non-leaky by construction:** folds hold out whole rulers, so the ruler one-hot is
all-zero at train time for the held-out ruler — the floor is "date an unseen ruler
from provenance + period", not a ruler->year lookup.

**Regimes:** `imbalanced` = all 1,193 labeled fragments; `balanced` = the same 200 MC
draws (168 frags, 8 rulers x 21) used by every other balanced result, mean/std over draws.

**Headline (year-raw):** balanced Spearman = 0.203 ±
0.083, MAE = 41 yr.
Compare against T2 (Ridge) / T1 (PLS) balanced Spearman per model.
