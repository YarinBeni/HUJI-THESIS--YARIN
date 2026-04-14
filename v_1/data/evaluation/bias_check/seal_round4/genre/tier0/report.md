# Bias Check — genre (tier0)

Generated: `2026-04-14T14:56:43Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `genre` |
| Cleaning | `tier0` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 384 |
| N classes (surviving) | 16 |
| Singletons dropped | 0 (none) |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.1 |
| CV accuracy | 0.6016 |
| CV macro-F1 | 0.3624 |
| CV weighted-F1 | 0.6098 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.3575 |
| Null mean | 0.0403 |
| Null std | 0.0119 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `lyrics` | 0.8824 |
| `incantations` | 0.7877 |
| `funerary texts` | 0.7692 |
| `rituals` | 0.6316 |
| `epics and myths` | 0.6018 |
| `literary letters` | 0.4348 |
| `hymns and prayers` | 0.4211 |
| `love literature` | 0.3226 |
| `prophecies` | 0.2857 |
| `epics` | 0.2353 |
| `chronicles` | 0.2222 |
| `lamentations` | 0.1379 |
| `wisdom literature` | 0.0667 |
| `catalogues` | 0.0000 |
| `commentary` | 0.0000 |
| `miscellaneous` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `catalogues` | `chronicles` | `commentary` | `epics` | `epics and myths` | `funerary texts` | `hymns and prayers` | `incantations` | `lamentations` | `literary letters` | `love literature` | `lyrics` | `miscellaneous` | `prophecies` | `rituals` | `wisdom literature` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `catalogues` | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `chronicles` | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `commentary` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| `epics` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 4 | 0 |
| `epics and myths` | 0 | 0 | 0 | 1 | 34 | 0 | 2 | 3 | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| `funerary texts` | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hymns and prayers` | 0 | 1 | 0 | 0 | 7 | 0 | 12 | 5 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| `incantations` | 0 | 0 | 0 | 1 | 3 | 2 | 11 | 115 | 9 | 0 | 3 | 0 | 6 | 2 | 2 | 7 |
| `lamentations` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `literary letters` | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 2 | 5 | 0 | 0 | 0 | 0 | 4 | 1 |
| `love literature` | 0 | 0 | 0 | 1 | 5 | 1 | 1 | 3 | 4 | 0 | 5 | 0 | 1 | 0 | 0 | 0 |
| `lyrics` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 | 2 | 0 |
| `miscellaneous` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `prophecies` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| `rituals` | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 18 | 0 |
| `wisdom literature` | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 |

