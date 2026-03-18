# Evaluation Corpus Size: 5,005 Texts (Letters Only)

**Date**: January 28, 2026
**Decision**: Use 5,005 letter texts for LLM baseline evaluation, not 6,577 total texts
**Status**: Validated by Nathan (researcher)

---

## Question

Why does the evaluation corpus contain **5,005 texts** instead of the expected **6,577 texts**?

---

## Answer

The difference of **1,572 texts** is due to filtering ORACC Group 2 data to **letters only** (`NALet`), excluding administrative documents (`NAAdm`).

### Breakdown by Source

| Source | Full Dataset | Letters Only | Removed |
|--------|--------------|--------------|---------|
| **ARCHIBAB** (Group 1 - Old Babylonian) | 1,526 | 1,526 | 0 |
| **ORACC** (Group 2 - Neo-Assyrian) | 4,007 | 2,435 | 1,572 (NAAdm) |
| **LBL** (Group 3 - Late Babylonian) | 1,044 | 1,044 | 0 |
| **TOTAL** | **6,577** | **5,005** | **1,572** |

### ORACC Domain Details

The ORACC corpus contains two domain types:
- **NALet**: Neo-Assyrian Letters (2,435 texts) ✅ Included
- **NAAdm**: Neo-Assyrian Administrative documents (1,572 texts) ❌ Excluded

---

## Justification

### 1. Genre Consistency (Primary Reason)

**Nathan's guidance** (email, January 2026):
> "We want our three corpora to be as much as possible even in their genre."

Including all 6,577 texts would create genre imbalance:
- **Group 1** (ARCHIBAB): All letters ✓
- **Group 2** (ORACC): Letters + Administrative texts ✗
- **Group 3** (LBL): All letters ✓

This imbalance would confound the temporal period predictions with genre differences.

### 2. ARCHIBAB Domains Are All Letters

**Nathan's clarification**:
> "The nine domains in ARCHIBAB are misleading: in fact they are all letters – of different content."

The ARCHIBAB corpus has 9 domain labels (administrative letter, political letter, private letter, etc.), but these are all **letter subtypes**, not different genres. The three "Unknown" variants are just uncertain/broken letters.

### 3. Corpus Validation

**Nathan's assessment**:
> "The three corpora seem to me to be smooth and ready for analysis: they are (mostly) composed of letters; their periodization is clear; their places of origin are also well separated."

He validated that the **5,005-text corpus** is appropriate for evaluation.

---

## Implementation

The filtering is implemented in [`v_1/src/evaluation/01_create_corpus.py`](../../v_1/src/evaluation/01_create_corpus.py) at lines 44-51:

```python
# STEP 3: Filter Group 2 (oracc) for letters only
letter_domain = [d for d in group2_full['domain'].unique() if 'let' in str(d).lower()]
filter_value = letter_domain[0] if letter_domain else 'NALet'
group2 = group2_full[group2_full['domain'] == filter_value].copy()

print(f"\n✓ Filtered Group 2 for letters (domain == '{filter_value}'):")
print(f"  - Before: {len(group2_full):,} words")
print(f"  - After: {len(group2):,} words")
print(f"  - Removed: {len(group2_full) - len(group2):,} words")
```

This filter selects only `NALet` domain from ORACC, removing `NAAdm`.

---

## Conclusion

The **5,005 text corpus** (further refined to **4,976** after domain label cleanup) is the correct size for evaluation. The original 6,577 count included administrative documents that would compromise genre consistency across temporal periods. This decision ensures:

1. **Genre homogeneity**: All three groups contain primarily letters
2. **Fair comparison**: Temporal predictions are not confounded by genre mixing
3. **Researcher validation**: Explicitly approved by domain expert (Nathan)
4. **Clean labels**: All texts have unambiguous domain labels (after subsequent cleanup)

---

---

## Subsequent Update: Domain Label Cleanup

**Date**: January 28, 2026
**Action**: Further cleanup to remove texts with Unknown/nan domain labels

After the initial filtering to 5,005 texts, an additional cleanup was performed to remove 29 texts with ambiguous or unknown domain labels:
- 22 texts with string "nan" in `domain_finegrained`
- 22 texts with "Other" in `domain_standard`
- 7 texts with "Unknown" domain labels

**Final corpus after domain cleanup**: **4,976 texts**

This additional 0.58% reduction ensures all texts have clear, unambiguous domain labels for reliable LLM evaluation metrics.

For detailed justification of this cleanup, see: [Domain Label Cleanup](domain_label_cleanup.md)

---

## ✅ Update: Final Corpus After March 8, 2026 Data Refresh

Chunrong delivered final cleaned versions of all three source CSV files on March 8, 2026. The corpus was rebuilt and the domain label cleanup re-applied. **Final corpus: 4,957 texts.**

| Group | Jan 2026 (prev. final) | Mar 2026 (current final) | Change |
|-------|------------------------|--------------------------|--------|
| Old Babylonian | 1,526 | **1,497** | −29 (domain cleanup) |
| Neo-Assyrian | 2,435 | **2,435** | 0 |
| Late Babylonian | 1,044 | **1,025** | −19 (bad texts deleted by Chunrong) |
| **Total** | **4,976** | **4,957** | **−19** |

All 5 verification checks passed (no Unknown/nan domain labels). This is the corpus used for LLM baseline evaluation.

**What changed in the new files:**
- ORACC: Aramaic entries removed, broken words removed, minor sign normalization (U4→UD, UNU→UNUG)
- LBL: Broken words removed, additional letters with bad transliteration deleted
- ARCHIBAB: Partially restored words handled per EvaCun paper standard, minor sign normalization

For full details of all cleaning decisions and justifications, see: [Chunrong Data Cleaning Decisions](chunrong_data_cleaning_decisions.md)

---

## Related Files

- Source data: [`v_1/data/processed/from_chungrong/`](../../v_1/data/processed/from_chungrong/)
- Preprocessing script: [`v_1/src/evaluation/01_create_corpus.py`](../../v_1/src/evaluation/01_create_corpus.py)
- Evaluation corpus: [`v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet`](../../v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet)
- Text preparation: [`v_1/src/evaluation/02_prepare_texts.py`](../../v_1/src/evaluation/02_prepare_texts.py)
- Domain cleanup notebook: [`v_1/notebooks/04_eda_evaluation.ipynb`](../../v_1/notebooks/04_eda_evaluation.ipynb) (Section 13-14)
- Data cleaning decisions: [chunrong_data_cleaning_decisions.md](chunrong_data_cleaning_decisions.md)
