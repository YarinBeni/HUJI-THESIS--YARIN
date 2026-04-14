# Bias Check — provenance (maximal)

Generated: `2026-04-14T15:08:52Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `provenance` |
| Cleaning | `maximal` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 374 |
| N classes (surviving) | 25 |
| Singletons dropped | 10 ('Assur;Nineveh', 'Babylon;Borsippa', 'Babylon;Sippar', 'Emar;Ugarit', 'Ešnunna' (+5 more)) |
| Effective k | 2 |

## CV Performance

| Metric | Value |
|--------|-------|
| Best C | 0.1 |
| CV accuracy | 0.2139 |
| CV macro-F1 | 0.1216 |
| CV weighted-F1 | 0.2133 |

## Permutation Test

| Field | Value |
|-------|-------|
| N permutations | 1000 |
| Actual macro-F1 | 0.1175 |
| Null mean | 0.0263 |
| Null std | 0.0099 |
| p-value | 0.0010 |
| Significance | **FAIL** |

## Per-Class F1

| Class | F1 |
|-------|----|
| `Babylon` | 0.6774 |
| `Nineveh` | 0.3636 |
| `Akhetaten` | 0.3077 |
| `Nippur` | 0.2500 |
| `Ugarit` | 0.1935 |
| `Susa` | 0.1818 |
| `Adab` | 0.1667 |
| `Šaduppûm` | 0.1667 |
| `Sippar` | 0.1379 |
| `Kaniš` | 0.1250 |
| `Larsa area` | 0.1250 |
| `Unknown` | 0.1129 |
| `Ur` | 0.0952 |
| `Isin` | 0.0800 |
| `Larsa` | 0.0556 |
| `Assur` | 0.0000 |
| `Dūr-Kurigalzu` | 0.0000 |
| `Emar` | 0.0000 |
| `Girsu` | 0.0000 |
| `Hattuša` | 0.0000 |
| `Kiš` | 0.0000 |
| `Mari` | 0.0000 |
| `Nerebtum` | 0.0000 |
| `Tell Duweihes` | 0.0000 |
| `Uruk` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Adab` | `Akhetaten` | `Assur` | `Babylon` | `Dūr-Kurigalzu` | `Emar` | `Girsu` | `Hattuša` | `Isin` | `Kaniš` | `Kiš` | `Larsa` | `Larsa area` | `Mari` | `Nerebtum` | `Nineveh` | `Nippur` | `Sippar` | `Susa` | `Tell Duweihes` | `Ugarit` | `Unknown` | `Ur` | `Uruk` | `Šaduppûm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Adab` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Akhetaten` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Assur` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `Babylon` | 1 | 0 | 0 | 42 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| `Dūr-Kurigalzu` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Emar` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Girsu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Hattuša` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `Isin` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Kaniš` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Kiš` | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| `Larsa` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| `Larsa area` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 3 | 1 | 2 | 3 | 3 | 4 | 3 | 1 | 0 | 0 | 0 | 2 | 2 | 1 | 0 | 1 | 1 |
| `Mari` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| `Nerebtum` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `Nineveh` | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| `Nippur` | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 2 | 0 | 0 | 1 | 1 | 1 | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| `Sippar` | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 3 | 1 | 0 | 1 | 0 | 0 | 0 |
| `Susa` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Tell Duweihes` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Ugarit` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 1 | 2 | 0 | 3 | 0 | 0 | 0 | 0 |
| `Unknown` | 3 | 4 | 2 | 7 | 5 | 1 | 1 | 10 | 8 | 5 | 4 | 11 | 8 | 4 | 5 | 1 | 1 | 1 | 5 | 4 | 6 | 7 | 3 | 1 | 5 |
| `Ur` | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `Uruk` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Šaduppûm` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |

