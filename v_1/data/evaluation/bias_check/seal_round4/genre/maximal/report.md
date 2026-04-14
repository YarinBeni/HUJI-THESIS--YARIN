# Bias Check — genre (maximal)

Generated: `2026-04-14T14:59:29Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `genre` |
| Cleaning | `maximal` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 384 |
| N classes (surviving) | 16 |
| Singletons dropped | 0 (none) |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.01 |
| CV accuracy | 0.3646 |
| CV macro-F1 | 0.2688 |
| CV weighted-F1 | 0.3995 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.2747 |
| Null mean | 0.0453 |
| Null std | 0.0124 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `lyrics` | 0.6875 |
| `epics` | 0.4706 |
| `epics and myths` | 0.4375 |
| `incantations` | 0.4151 |
| `hymns and prayers` | 0.4068 |
| `literary letters` | 0.4000 |
| `rituals` | 0.3793 |
| `love literature` | 0.3478 |
| `funerary texts` | 0.3333 |
| `prophecies` | 0.1667 |
| `wisdom literature` | 0.1455 |
| `lamentations` | 0.1111 |
| `catalogues` | 0.0000 |
| `chronicles` | 0.0000 |
| `commentary` | 0.0000 |
| `miscellaneous` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `catalogues` | `chronicles` | `commentary` | `epics` | `epics and myths` | `funerary texts` | `hymns and prayers` | `incantations` | `lamentations` | `literary letters` | `love literature` | `lyrics` | `miscellaneous` | `prophecies` | `rituals` | `wisdom literature` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `catalogues` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `chronicles` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| `commentary` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 |
| `epics` | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 0 |
| `epics and myths` | 0 | 0 | 1 | 0 | 21 | 1 | 1 | 3 | 7 | 2 | 1 | 2 | 2 | 1 | 0 | 6 |
| `funerary texts` | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hymns and prayers` | 0 | 0 | 1 | 0 | 2 | 1 | 12 | 2 | 2 | 0 | 3 | 0 | 2 | 2 | 0 | 1 |
| `incantations` | 8 | 3 | 0 | 0 | 15 | 6 | 16 | 44 | 11 | 8 | 8 | 3 | 8 | 4 | 6 | 21 |
| `lamentations` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `literary letters` | 0 | 2 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `love literature` | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 8 | 1 | 0 | 0 | 0 | 5 |
| `lyrics` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 22 | 0 | 0 | 8 | 0 |
| `miscellaneous` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `prophecies` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `rituals` | 0 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 11 | 0 |
| `wisdom literature` | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 2 | 2 | 2 | 2 | 0 | 1 | 0 | 0 | 4 |

