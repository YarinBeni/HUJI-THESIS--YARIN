# Bias Check — period (tier0)

Generated: `2026-04-14T15:15:19Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `period` |
| Cleaning | `tier0` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 383 |
| N classes (surviving) | 9 |
| Singletons dropped | 1 ('Later Periods (SB, NA, LB)') |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.1 |
| CV accuracy | 0.7572 |
| CV macro-F1 | 0.4732 |
| CV weighted-F1 | 0.7511 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.4726 |
| Null mean | 0.0934 |
| Null std | 0.0183 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `Late Babylonian` | 0.8941 |
| `Old Babylonian` | 0.8659 |
| `Neo or Late Babylonian` | 0.7368 |
| `Middle Babylonian/Assyrian` | 0.6567 |
| `Old Assyrian` | 0.5455 |
| `Neo-Assyrian` | 0.4828 |
| `Middle Babylonian` | 0.0769 |
| `Archaic/Old Akkadian/Ebla` | 0.0000 |
| `Middle Assyrian` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Archaic/Old Akkadian/Ebla` | `Late Babylonian` | `Middle Assyrian` | `Middle Babylonian` | `Middle Babylonian/Assyrian` | `Neo or Late Babylonian` | `Neo-Assyrian` | `Old Assyrian` | `Old Babylonian` |
|---|---|---|---|---|---|---|---|---|---|
| `Archaic/Old Akkadian/Ebla` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `Late Babylonian` | 0 | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Middle Assyrian` | 0 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| `Middle Babylonian` | 0 | 0 | 0 | 2 | 3 | 0 | 0 | 0 | 19 |
| `Middle Babylonian/Assyrian` | 0 | 3 | 2 | 2 | 22 | 0 | 0 | 0 | 6 |
| `Neo or Late Babylonian` | 0 | 1 | 0 | 0 | 0 | 21 | 4 | 0 | 0 |
| `Neo-Assyrian` | 0 | 1 | 0 | 0 | 0 | 10 | 7 | 0 | 0 |
| `Old Assyrian` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 |
| `Old Babylonian` | 0 | 1 | 0 | 24 | 5 | 0 | 0 | 2 | 197 |

