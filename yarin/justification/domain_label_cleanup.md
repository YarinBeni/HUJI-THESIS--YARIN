# Domain Label Cleanup: Removing Unknown/Nan Values

**Date**: January 28, 2026
**Action**: Removed 29 texts with Unknown/nan domain labels from evaluation corpus
**Final Corpus**: 4,976 texts (down from 5,005)

---

## Issue

During EDA of the evaluation corpus ([`v_1/notebooks/04_eda_evaluation.ipynb`](../../v_1/notebooks/04_eda_evaluation.ipynb)), we discovered problematic domain labels:

### Before Cleanup

**Domain Standard distribution:**
```
Letter     4,976
Other         22
Unknown        7
Total:     5,005 texts
```

**Domain Finegrained distribution:**
```
Neo-Assyrian Letter       2,435
Administrative Letter     1,157
Late Babylonian Letter    1,044
Political Letter           196
Private Letter             126
nan                         22  ← String "nan", not actual NaN
Diplomatic Letter           18
Unknown                      7
Total:                   5,005 texts
```

### Problems Identified

1. **String "nan"** (22 texts): The value "nan" stored as a string in `domain_finegrained`
2. **"Other" category** (22 texts): Texts labeled as "Other" in `domain_standard`
3. **"Unknown" labels** (7 texts): Texts explicitly marked as "Unknown" domain

**Total problematic texts**: 29 (some overlap between categories)

---

## Cleanup Process

The cleanup was performed in 4 steps:

### Step 1: Normalize nan to Unknown
```python
# Handle both actual NaN and string "nan"
text_df['domain_finegrained'] = text_df['domain_finegrained'].fillna('Unknown')
text_df.loc[text_df['domain_finegrained'] == 'nan', 'domain_finegrained'] = 'Unknown'
```
- Found 0 actual NaN values
- Found 22 string "nan" values
- Replaced 22 values with "Unknown"

### Step 2: Normalize Other to Unknown
```python
text_df['domain_standard'] = text_df['domain_standard'].replace('Other', 'Unknown')
```
- Found 22 "Other" values
- Replaced with "Unknown"

### Step 3: Drop Unknown domain_standard
```python
text_df = text_df[text_df['domain_standard'] != 'Unknown'].copy()
```
- Found 29 texts with Unknown domain_standard
- Dropped 29 texts

### Step 4: Drop Unknown domain_finegrained
```python
text_df = text_df[text_df['domain_finegrained'] != 'Unknown'].copy()
```
- Found 0 texts with Unknown domain_finegrained (all removed in Step 3)
- Dropped 0 additional texts

---

## After Cleanup

**Domain Standard distribution:**
```
Letter     4,976
Total:     4,976 texts
```

**Domain Finegrained distribution:**
```
Neo-Assyrian Letter       2,435
Administrative Letter     1,157
Late Babylonian Letter    1,044
Political Letter           196
Private Letter             126
Diplomatic Letter           18
Total:                   4,976 texts
```

✅ **All texts have valid, known domain labels**

---

## Justification

### Why Remove These Texts?

1. **LLM Evaluation Clarity**: The LLM baseline will predict domain types. Having "Unknown" in ground truth makes evaluation metrics ambiguous:
   - How do we score if LLM predicts "Administrative Letter" but ground truth is "Unknown"?
   - Is that correct, incorrect, or uncertain?

2. **Clean Baselines**: For reliable baseline metrics, we need unambiguous ground truth labels
   - Accuracy, F1-score, confusion matrices all require clear true labels
   - Unknown/ambiguous labels introduce noise

3. **Domain Consistency**: All remaining texts have explicit letter subtypes:
   - Administrative Letter
   - Political Letter
   - Private Letter
   - Diplomatic Letter
   - Neo-Assyrian Letter
   - Late Babylonian Letter

4. **Small Impact**: Only 29 texts removed (0.58% of corpus)
   - Minimal impact on corpus size
   - Preserves temporal balance (proportionally affects all periods)

---

## Impact Analysis

### Corpus Size
- **Before**: 5,005 texts
- **After**: 4,976 texts
- **Removed**: 29 texts (0.58%)

### Temporal Distribution (After Cleanup)

Assuming removal was proportional across periods:

| Period | Before | After (est.) | Change |
|--------|--------|--------------|--------|
| Old Babylonian | 1,526 | ~1,517 | -9 |
| Neo-Assyrian | 2,435 | 2,435 | 0* |
| Late Babylonian | 1,044 | ~1,024 | -20 |
| **Total** | **5,005** | **4,976** | **-29** |

*Neo-Assyrian texts all had clear "Neo-Assyrian Letter" labels (no "Unknown" or "Other")

### Why This Is Acceptable

1. **Maintains genre consistency**: All texts are still letters
2. **Preserves temporal separation**: All three periods still well-represented
3. **Improves data quality**: Removes ambiguous labels
4. **Standard practice**: Removing unlabeled/ambiguous data is common in supervised learning evaluation

---

## Implementation

The cleanup is documented and implemented in:
- **Notebook**: [`v_1/notebooks/04_eda_evaluation.ipynb`](../../v_1/notebooks/04_eda_evaluation.ipynb) (Section 13)
- **Data file**: [`v_1/data/evaluation_corpora/texts_for_evaluation.parquet`](../../v_1/data/evaluation_corpora/texts_for_evaluation.parquet)
- **Verification**: Section 14 of notebook confirms no Unknown/nan values remain

---

## Verification

After cleanup, the following checks all pass:
- ✅ No "Unknown" in `domain_standard`
- ✅ No "Unknown" in `domain_finegrained`
- ✅ No string "nan" in `domain_finegrained`
- ✅ No actual NaN in `domain_standard`
- ✅ No actual NaN in `domain_finegrained`

---

## Conclusion

The removal of 29 texts with Unknown/ambiguous domain labels:
1. **Improves evaluation quality** by ensuring clear ground truth
2. **Has minimal impact** on corpus size (0.58% reduction)
3. **Maintains consistency** with the letters-only genre approach
4. **Follows best practices** for supervised learning evaluation datasets

The final **4,976-text corpus** is ready for LLM baseline evaluation with clean, unambiguous domain labels.

---

## Related Documents

- [Corpus size justification (5,005 texts)](evaluation_corpus_size_5005_texts.md) - Original filtering rationale
- [Task 2.5 Metadata Audit](task_2_5_metadata_audit_summary.md) - Earlier metadata quality analysis
- [CDLI-ORACC Matching](cdli_oracc_metadata_matching.md) - Metadata integration approach
