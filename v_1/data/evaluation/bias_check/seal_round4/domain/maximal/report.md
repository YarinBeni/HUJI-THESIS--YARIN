# Bias Check — domain (maximal)

Generated: `2026-04-14T15:52:01Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `domain` |
| Cleaning | `maximal` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 384 |
| N classes (surviving) | 3 |
| Singletons dropped | 0 (none) |
| Effective k | 5 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.01 |
| CV accuracy | 0.9479 |
| CV macro-F1 | 0.8761 |
| CV weighted-F1 | 0.9471 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.8744 |
| Null mean | 0.3312 |
| Null std | 0.0281 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `SEAL` | 0.9803 |
| `DLL` | 0.8372 |
| `LBPL` | 0.8108 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `DLL` | `LBPL` | `SEAL` |
|---|---|---|---|
| `DLL` | 36 | 2 | 6 |
| `LBPL` | 6 | 30 | 2 |
| `SEAL` | 0 | 4 | 298 |

