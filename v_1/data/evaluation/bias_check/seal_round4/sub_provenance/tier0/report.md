# Bias Check — sub_provenance (tier0)

Generated: `2026-04-14T16:05:29Z`
Script: `v_1/src/bias_check/06_bias_check_cv.py`

## Result: FAIL

**FAIL (p < 0.01 — significant bias detected)**

## Data

| Field | Value |
|-------|-------|
| Task | `sub_provenance` |
| Cleaning | `tier0` |
| Corpora pooled | seal, dll, lbpl |
| N fragments | 374 |
| N classes (surviving) | 25 |
| Singletons dropped | 10 ('Unknown;mod. Tell es-Senkereh', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;mod. Tell Abu Ḥabbah', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;modern Birs Nimrud', 'mod. Qalʿat Sharqat;mod. Kouyunjik, Tell Nabi Yunus', 'mod. Tell Abu Ḥabbah;mod. Nuffar' (+5 more)) |
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
| `mod. Tell el-Amarna` | 1.0000 |
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0.6718 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0.4231 |
| `mod. Shush` | 0.3125 |
| `mod. Kültepe` | 0.2857 |
| `mod. Tell el-Uhaymir` | 0.2857 |
| `mod. Tell Ḥarmal` | 0.2667 |
| `mod. Ras Shamrah` | 0.2400 |
| `Larsa area` | 0.2195 |
| `mod. Nuffar` | 0.2000 |
| `Unknown` | 0.1343 |
| `mod. Tell es-Senkereh` | 0.0851 |
| `mod. Tell Ḥariri` | 0.0833 |
| `mod. Tell Abu Ḥabbah` | 0.0741 |
| `mod. Boghazköy` | 0.0000 |
| `mod. Ishan Baḥriyat` | 0.0000 |
| `mod. Ishchali` | 0.0000 |
| `mod. Qalʿat Sharqat` | 0.0000 |
| `mod. Tell Bismaya` | 0.0000 |
| `mod. Tell Meskene` | 0.0000 |
| `mod. Tell el-Muqayyar` | 0.0000 |
| `mod. Telloh` | 0.0000 |
| `mod. Warka` | 0.0000 |
| `mod. ʿAqar Quf` | 0.0000 |
| `vicinity of Nippur` | 0.0000 |

## Confusion Matrix

Rows = true label, columns = predicted label.

| True \ Pred | `Larsa area` | `Unknown` | `mod. Boghazköy` | `mod. Ishan Baḥriyat` | `mod. Ishchali` | `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | `mod. Kouyunjik, Tell Nabi Yunus` | `mod. Kültepe` | `mod. Nuffar` | `mod. Qalʿat Sharqat` | `mod. Ras Shamrah` | `mod. Shush` | `mod. Tell Abu Ḥabbah` | `mod. Tell Bismaya` | `mod. Tell Meskene` | `mod. Tell el-Amarna` | `mod. Tell el-Muqayyar` | `mod. Tell el-Uhaymir` | `mod. Tell es-Senkereh` | `mod. Tell Ḥariri` | `mod. Tell Ḥarmal` | `mod. Telloh` | `mod. Warka` | `mod. ʿAqar Quf` | `vicinity of Nippur` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Larsa area` | 9 | 4 | 2 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 3 | 2 | 4 | 0 | 0 | 0 | 0 | 0 |
| `Unknown` | 25 | 9 | 16 | 3 | 2 | 7 | 2 | 1 | 2 | 1 | 3 | 7 | 3 | 1 | 0 | 0 | 3 | 2 | 15 | 6 | 4 | 0 | 0 | 0 | 0 |
| `mod. Boghazköy` | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishan Baḥriyat` | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishchali` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0 | 0 | 3 | 0 | 0 | 44 | 11 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0 | 0 | 0 | 0 | 0 | 8 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Kültepe` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Nuffar` | 2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Qalʿat Sharqat` | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ras Shamrah` | 0 | 0 | 2 | 0 | 0 | 4 | 3 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Shush` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Abu Ḥabbah` | 2 | 1 | 0 | 1 | 0 | 3 | 4 | 0 | 2 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Bismaya` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Meskene` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Amarna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Muqayyar` | 2 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Uhaymir` | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell es-Senkereh` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Ḥariri` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| `mod. Tell Ḥarmal` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| `mod. Telloh` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Warka` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. ʿAqar Quf` | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `vicinity of Nippur` | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

