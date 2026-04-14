# Bias Check — domain (tier0)

Generated: `2026-04-14T15:46:17Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `domain` |
| Cleaning | `tier0` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 384 |
| N classes (surviving) | 3 |
| Singletons dropped | 0 (none) |
| Effective k | 5 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 10.0 |
| CV accuracy | 0.9792 |
| CV macro-F1 | 0.9518 |
| CV weighted-F1 | 0.9789 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.9506 |
| Null mean | 0.3115 |
| Null std | 0.0206 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `SEAL` | 0.9918 |
| `LBPL` | 0.9333 |
| `DLL` | 0.9302 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `DLL` | `LBPL` | `SEAL` |
|---|---|---|---|
| `DLL` | 40 | 1 | 3 |
| `LBPL` | 2 | 35 | 1 |
| `SEAL` | 0 | 1 | 301 |

