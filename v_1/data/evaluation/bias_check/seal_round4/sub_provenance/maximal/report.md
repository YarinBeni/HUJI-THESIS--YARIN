# Bias Check — sub_provenance (maximal)

Generated: `2026-04-14T16:08:46Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `sub_provenance` |
| Cleaning | `maximal` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 374 |
| N classes (surviving) | 25 |
| Singletons dropped | 10 ('Unknown;mod. Tell es-Senkereh', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;mod. Tell Abu Ḥabbah', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;modern Birs Nimrud', 'mod. Qalʿat Sharqat;mod. Kouyunjik, Tell Nabi Yunus', 'mod. Tell Abu Ḥabbah;mod. Nuffar' (+5 more)) |
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
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0.6774 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0.3636 |
| `mod. Tell el-Amarna` | 0.3077 |
| `mod. Nuffar` | 0.2500 |
| `mod. Ras Shamrah` | 0.1935 |
| `mod. Shush` | 0.1818 |
| `mod. Tell Bismaya` | 0.1667 |
| `mod. Tell Ḥarmal` | 0.1667 |
| `mod. Tell Abu Ḥabbah` | 0.1379 |
| `Larsa area` | 0.1250 |
| `mod. Kültepe` | 0.1250 |
| `Unknown` | 0.1129 |
| `mod. Tell el-Muqayyar` | 0.0952 |
| `mod. Ishan Baḥriyat` | 0.0800 |
| `mod. Tell es-Senkereh` | 0.0556 |
| `mod. Boghazköy` | 0.0000 |
| `mod. Ishchali` | 0.0000 |
| `mod. Qalʿat Sharqat` | 0.0000 |
| `mod. Tell Meskene` | 0.0000 |
| `mod. Tell el-Uhaymir` | 0.0000 |
| `mod. Tell Ḥariri` | 0.0000 |
| `mod. Telloh` | 0.0000 |
| `mod. Warka` | 0.0000 |
| `mod. ʿAqar Quf` | 0.0000 |
| `vicinity of Nippur` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Larsa area` | `Unknown` | `mod. Boghazköy` | `mod. Ishan Baḥriyat` | `mod. Ishchali` | `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | `mod. Kouyunjik, Tell Nabi Yunus` | `mod. Kültepe` | `mod. Nuffar` | `mod. Qalʿat Sharqat` | `mod. Ras Shamrah` | `mod. Shush` | `mod. Tell Abu Ḥabbah` | `mod. Tell Bismaya` | `mod. Tell Meskene` | `mod. Tell el-Amarna` | `mod. Tell el-Muqayyar` | `mod. Tell el-Uhaymir` | `mod. Tell es-Senkereh` | `mod. Tell Ḥariri` | `mod. Tell Ḥarmal` | `mod. Telloh` | `mod. Warka` | `mod. ʿAqar Quf` | `vicinity of Nippur` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Larsa area` | 3 | 1 | 3 | 3 | 3 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 3 | 4 | 1 | 0 | 1 | 1 | 2 |
| `Unknown` | 8 | 7 | 10 | 8 | 5 | 7 | 1 | 5 | 1 | 2 | 6 | 5 | 1 | 3 | 1 | 4 | 3 | 4 | 11 | 4 | 5 | 1 | 1 | 5 | 4 |
| `mod. Boghazköy` | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| `mod. Ishan Baḥriyat` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishchali` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0 | 0 | 3 | 0 | 0 | 42 | 5 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 3 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0 | 0 | 0 | 1 | 0 | 6 | 8 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `mod. Kültepe` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Nuffar` | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 2 | 4 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| `mod. Qalʿat Sharqat` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ras Shamrah` | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Shush` | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Abu Ḥabbah` | 0 | 1 | 0 | 1 | 0 | 4 | 4 | 0 | 0 | 1 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Tell Bismaya` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `mod. Tell Meskene` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Amarna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Muqayyar` | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| `mod. Tell el-Uhaymir` | 0 | 1 | 2 | 1 | 0 | 1 | 1 | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell es-Senkereh` | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| `mod. Tell Ḥariri` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Tell Ḥarmal` | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| `mod. Telloh` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Warka` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. ʿAqar Quf` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `vicinity of Nippur` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |

