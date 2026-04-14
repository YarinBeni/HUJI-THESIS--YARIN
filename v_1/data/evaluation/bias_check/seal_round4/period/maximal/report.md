# Bias Check — period (maximal)

Generated: `2026-04-14T14:46:59Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `period` |
| Cleaning | `maximal` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 383 |
| N classes (surviving) | 9 |
| Singletons dropped | 1 ('Later Periods (SB, NA, LB)') |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.001 |
| CV accuracy | 0.5901 |
| CV macro-F1 | 0.3522 |
| CV weighted-F1 | 0.6210 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.3498 |
| Null mean | 0.0834 |
| Null std | 0.0198 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `Old Babylonian` | 0.7452 |
| `Late Babylonian` | 0.6988 |
| `Neo or Late Babylonian` | 0.6452 |
| `Middle Babylonian/Assyrian` | 0.3922 |
| `Neo-Assyrian` | 0.3226 |
| `Old Assyrian` | 0.1481 |
| `Middle Babylonian` | 0.1270 |
| `Middle Assyrian` | 0.0909 |
| `Archaic/Old Akkadian/Ebla` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Archaic/Old Akkadian/Ebla` | `Late Babylonian` | `Middle Assyrian` | `Middle Babylonian` | `Middle Babylonian/Assyrian` | `Neo or Late Babylonian` | `Neo-Assyrian` | `Old Assyrian` | `Old Babylonian` |
|---|---|---|---|---|---|---|---|---|---|
| `Archaic/Old Akkadian/Ebla` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `Late Babylonian` | 0 | 29 | 3 | 0 | 0 | 5 | 0 | 0 | 1 |
| `Middle Assyrian` | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 2 |
| `Middle Babylonian` | 0 | 0 | 2 | 4 | 2 | 0 | 0 | 1 | 15 |
| `Middle Babylonian/Assyrian` | 0 | 6 | 4 | 3 | 10 | 0 | 3 | 0 | 9 |
| `Neo or Late Babylonian` | 0 | 3 | 0 | 0 | 0 | 20 | 3 | 0 | 0 |
| `Neo-Assyrian` | 0 | 2 | 0 | 0 | 0 | 10 | 5 | 1 | 0 |
| `Old Assyrian` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| `Old Babylonian` | 9 | 4 | 6 | 32 | 3 | 0 | 2 | 18 | 155 |

