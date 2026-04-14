# Bias Check — provenance (tier0)

Generated: `2026-04-14T15:05:05Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `provenance` |
| Cleaning | `tier0` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 374 |
| N classes (surviving) | 25 |
| Singletons dropped | 10 ('Assur;Nineveh', 'Babylon;Borsippa', 'Babylon;Sippar', 'Emar;Ugarit', 'Ešnunna' (+5 more)) |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 1.0 |
| CV accuracy | 0.2594 |
| CV macro-F1 | 0.1713 |
| CV weighted-F1 | 0.2431 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.1672 |
| Null mean | 0.0305 |
| Null std | 0.0098 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `Akhetaten` | 1.0000 |
| `Babylon` | 0.6718 |
| `Nineveh` | 0.4231 |
| `Susa` | 0.3125 |
| `Kaniš` | 0.2857 |
| `Kiš` | 0.2857 |
| `Šaduppûm` | 0.2667 |
| `Ugarit` | 0.2400 |
| `Larsa area` | 0.2195 |
| `Nippur` | 0.2000 |
| `Unknown` | 0.1343 |
| `Larsa` | 0.0851 |
| `Mari` | 0.0833 |
| `Sippar` | 0.0741 |
| `Adab` | 0.0000 |
| `Assur` | 0.0000 |
| `Dūr-Kurigalzu` | 0.0000 |
| `Emar` | 0.0000 |
| `Girsu` | 0.0000 |
| `Hattuša` | 0.0000 |
| `Isin` | 0.0000 |
| `Nerebtum` | 0.0000 |
| `Tell Duweihes` | 0.0000 |
| `Ur` | 0.0000 |
| `Uruk` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Adab` | `Akhetaten` | `Assur` | `Babylon` | `Dūr-Kurigalzu` | `Emar` | `Girsu` | `Hattuša` | `Isin` | `Kaniš` | `Kiš` | `Larsa` | `Larsa area` | `Mari` | `Nerebtum` | `Nineveh` | `Nippur` | `Sippar` | `Susa` | `Tell Duweihes` | `Ugarit` | `Unknown` | `Ur` | `Uruk` | `Šaduppûm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Adab` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Akhetaten` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Assur` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Babylon` | 0 | 0 | 0 | 44 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Dūr-Kurigalzu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Emar` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Girsu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Hattuša` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `Isin` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `Kaniš` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Kiš` | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 4 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Larsa` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 1 | 2 | 0 | 0 |
| `Larsa area` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 3 | 2 | 9 | 4 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 4 | 1 | 0 | 0 |
| `Mari` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Nerebtum` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `Nineveh` | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Nippur` | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 2 | 1 | 1 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| `Sippar` | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 2 | 0 | 0 | 4 | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Susa` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Tell Duweihes` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Ugarit` | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| `Unknown` | 1 | 0 | 1 | 7 | 0 | 0 | 0 | 16 | 3 | 1 | 2 | 15 | 25 | 6 | 2 | 2 | 2 | 3 | 7 | 0 | 3 | 9 | 3 | 0 | 4 |
| `Ur` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 2 | 0 | 0 | 0 |
| `Uruk` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `Šaduppûm` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

