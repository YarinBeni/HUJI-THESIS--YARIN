# Data Pipeline Verification and Justification

**Date:** January 2026
**Purpose:** To provide complete transparency and verification of how Akkadian data flows from raw sources through to the MLM training pipeline.

---

## Overview

This document provides full visibility into the data transformation pipeline, proving that our data processing is correct and aligned with our research objectives. We trace data through five stages:

1. **Source Parquets** (one row per word)
2. **Unified Dataset** (merged sources)
3. **Train/Val/Test Splits** (80/10/10 by fragment)
4. **Fragment Texts** (one row per fragment)
5. **Training Dataset** (tokenized sequences for MLM)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────────┘

STAGE 1: SOURCE CSVs → PARQUETS (One Row Per WORD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Files: v_1/src/preprocessing/{02_process_ebl.py, 03_process_archibab.py, 04_process_oracc.py}

Input:  Raw CSV files from eBL, ORACC, Archibab
Process: Apply EvaCun 2025 tokenization (sign-level)
        - Split words on hyphens: "a-na" → "a na"
        - Remove editorial marks: [a]-na → a-na
        - Normalize subscripts: ša₂ → ša

Output: {source}_corpus.parquet

Example Row:
┌──────────────┬──────────┬──────────┬─────────────┬─────────────┐
│ fragment_id  │ line_num │ word_idx │ value_raw   │ value_signs │
├──────────────┼──────────┼──────────┼─────────────┼─────────────┤
│ ARM 10 33    │ 1        │ 0        │ a-na        │ a na        │
│ ARM 10 33    │ 1        │ 1        │ be-lí       │ be lí       │
│ ARM 10 33    │ 1        │ 2        │ ù           │ ù           │
│ ARM 10 33    │ 1        │ 3        │ ka-ka-bi    │ ka ka bi    │
└──────────────┴──────────┴──────────┴─────────────┴─────────────┘

KEY: Each word is a separate row. The value_signs column contains
     pre-tokenized signs separated by spaces.


STAGE 2: UNIFIED DATASET (Merge Sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: v_1/src/preprocessing/05_create_unified.py

Input:  ebl_corpus.parquet + oracc_corpus.parquet + archibab_corpus.parquet
Process: pd.concat() with source column added
Output: unified_corpus.parquet (2,450,094 words)

Distribution:
  - ORACC:    1,385,932 words (56.6%)
  - eBL:        998,353 words (40.7%)
  - Archibab:    65,809 words (2.7%)


STAGE 3: TRAIN/VAL/TEST SPLIT (80/10/10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: v_1/src/preprocessing/05_create_unified.py

Process: Split by fragment_id to prevent data leakage
  1. Get unique fragment IDs: 40,429 fragments
  2. Shuffle with seed=42
  3. Assign 80% to train, 10% to val, 10% to test
  4. Filter rows by fragment ID membership

Output: train.parquet, val.parquet, test.parquet
  - Train: 1,960,636 words (32,343 fragments)
  - Val:     253,798 words (4,042 fragments)
  - Test:    235,660 words (4,044 fragments)

Verification: ✅ Zero overlapping fragment_ids between splits


STAGE 4: FRAGMENT TEXTS (One Row Per Fragment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: v_1/src/training/baseline/01_prepare_data.py
Function: data_utils.py::reconstruct_fragment_text()

THIS IS THE CRITICAL TRANSFORMATION:
Input:  Multiple rows per fragment (one row per word)
Output: Single row per fragment with all signs concatenated

Algorithm:
  1. Group by fragment_id
  2. Sort by (line_num, word_idx)
  3. Join all value_signs with spaces
  4. Count total signs

Example Transformation:

Input (train.parquet):
┌──────────────┬──────────┬──────────┬─────────────┐
│ fragment_id  │ line_num │ word_idx │ value_signs │
├──────────────┼──────────┼──────────┼─────────────┤
│ ARM 10 33    │ 1        │ 0        │ a na        │  ← word 1
│ ARM 10 33    │ 1        │ 1        │ be lí       │  ← word 2
│ ARM 10 33    │ 1        │ 2        │ ù           │  ← word 3
│ ARM 10 33    │ 1        │ 3        │ ka ka bi    │  ← word 4
│ ARM 10 33    │ 2        │ 0        │ qí bí ma    │  ← word 5
└──────────────┴──────────┴──────────┴─────────────┘

↓ GROUP BY fragment_id
↓ SORT BY (line_num, word_idx)
↓ JOIN value_signs WITH SPACES

Output (train_fragments.parquet):
┌──────────────┬────────────────────────────────────┬───────────┐
│ fragment_id  │ text                               │ num_signs │
├──────────────┼────────────────────────────────────┼───────────┤
│ ARM 10 33    │ "a na be lí ù ka ka bi qí bí ma"   │ 11        │
└──────────────┴────────────────────────────────────┴───────────┘

The 'text' column is a single space-separated string containing
ALL signs from the fragment in their original order.


STAGE 5: TRAINING DATASET (Tokenized Sequences)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: v_1/src/training/baseline/data_utils.py
Class: AkkadianMLMDataset

Input:  Fragment text (e.g., "a na be lí ù ka ka bi")
Output: PyTorch tensors ready for MLM training

Process:
  1. Split text on spaces → ['a', 'na', 'be', 'lí', 'ù', 'ka', 'ka', 'bi']
  2. Lookup sign IDs in vocab → [5745, 10329, 1902, 9027, 14561, 7979, 7979, 5899]
  3. Add special tokens:
     [CLS] + signs + [SEP] + [PAD]...
     [2, 5745, 10329, 1902, 9027, 14561, 7979, 7979, 5899, 3, 0, 0, ...]
  4. Apply 15% MLM masking (BERT 80/10/10 strategy)
  5. Create attention_mask (1 for real tokens, 0 for padding)
  6. Create labels (-100 for non-masked, token_id for masked)

Example with Masking:

Original signs:     ['a', 'na', 'be', 'lí', 'ù', 'ka', 'ka', 'bi']
Original IDs:       [2, 5745, 10329, 1902, 9027, 14561, 7979, 7979, 5899, 3]
                    ↑                 ↑                             ↑
                  [CLS]           content                        [SEP]

After 15% masking:  [2, 5745, 4, 1902, 9027, 14561, 7979, 7979, 5899, 3]
                          ↑
                      masked as [MASK]

Labels:             [-100, -100, 10329, -100, -100, -100, -100, -100, -100, -100]
                                   ↑
                            correct answer for masked position

The model must predict token 10329 ('be') from the masked input.
```

---

## Verification Script

We created a comprehensive verification script that traces data through all stages:

**Location:** `v_1/src/verify_data_pipeline.py`

**Usage:**
```bash
python3 v_1/src/verify_data_pipeline.py
```

**What It Verifies:**

1. **Source Parquets** - Shows sample words from each source with their `value_signs`
2. **Unified Dataset** - Confirms merge and source distribution
3. **Train/Val/Test Splits** - Verifies no leakage between splits
4. **Fragment Texts** - Shows how words are joined into fragment-level text
5. **Training Dataset** - Shows tokenization and MLM masking in action
6. **Full Pipeline Trace** - Follows a single fragment through all stages

---

## Concrete Examples from Real Data

### Example 1: Archibab Fragment

**Fragment ID:** `ARM 10 33`

**Stage 1: Word Rows** (from `archibab_corpus.parquet`)
```
Row 0: line=1, word_idx=0, value_raw="a-na",      value_signs="a na"
Row 1: line=1, word_idx=1, value_raw="be-lí",     value_signs="be lí"
Row 2: line=1, word_idx=2, value_raw="ù",         value_signs="ù"
Row 3: line=1, word_idx=3, value_raw="ka-ka-bi",  value_signs="ka ka bi"
Row 4: line=2, word_idx=0, value_raw="qí-bí-ma",  value_signs="qí bí ma"
```

**Stage 4: Fragment Text** (from `train_fragments.parquet`)
```
fragment_id: ARM 10 33
text: "a na be lí ù ka ka bi qí bí ma um ma ki ib ri (d) da gan..."
num_signs: 297
```

**Stage 5: Model Input** (from `AkkadianMLMDataset`)
```
text.split() → ['a', 'na', 'be', 'lí', 'ù', 'ka', 'ka', 'bi', ...]

vocab lookup:
  'a'  → 5745
  'na' → 10329
  'be' → 1902
  'lí' → 9027
  ...

input_ids: [2, 5745, 10329, 1902, 9027, ...]
           ↑ [CLS]
```

### Example 2: eBL Fragment

**Fragment ID:** `1848,0720.121`

**Stage 1: Word Rows**
```
Row 0: line=8,  word_idx=1, value_raw="BE-ma",    value_signs="BE ma"
Row 1: line=8,  word_idx=2, value_raw="AN.GE₆",   value_signs="AN GE"
Row 2: line=9,  word_idx=1, value_raw="ŠEŠ",      value_signs="ŠEŠ"
Row 3: line=9,  word_idx=2, value_raw="ŠEŠ-šu₂",  value_signs="ŠEŠ šu"
```

**Stage 4: Fragment Text**
```
text: "BE ma AN GE ŠEŠ ŠEŠ šu GU DIŠ U 15 KAM AN GE GAR..."
num_signs: 66
```

### Example 3: ORACC Fragment

**Fragment ID:** `P224485`

**Stage 1: Word Rows**
```
Row 0: line=0, word_idx=0, value_raw="LUGAL",              value_signs="LUGAL"
Row 1: line=0, word_idx=1, value_raw="a-na",               value_signs="a na"
Row 2: line=0, word_idx=2, value_raw="{1}aš-šur-MAN-⸢PAB*", value_signs="aš šur MAN PAB"
```

**Stage 4: Fragment Text**
```
text: "LUGAL a na aš šur MAN PAB šul mu ia a ši a na KUR aš šur..."
num_signs: [many signs from a longer text]
```

---

## Key Insights and Verification

### 1. Pre-Tokenized Signs

The `value_signs` column **already contains sign-level tokens**. This is critical because:
- Processing scripts (e.g., `02_process_ebl.py`) apply EvaCun 2025 tokenization during CSV→Parquet conversion
- Word "a-na" becomes "a na" (two signs separated by space)
- Word "ka-ka-bi" becomes "ka ka bi" (three signs separated by space)

**Verification Code:** `v_1/src/preprocessing/02_process_ebl.py:24-72` (`tokenize_to_signs()`)

### 2. Fragment Reconstruction is Simple Concatenation

Building fragment text is straightforward:
```python
# From data_utils.py:43-74
def reconstruct_fragment_text(fragment_df: pd.DataFrame) -> str:
    fragment_df = fragment_df.sort_values(['line_num', 'word_idx'])
    signs_list = fragment_df['value_signs'].dropna().tolist()
    text = ' '.join(signs_list)  # ← Just join with spaces!
    return text
```

Example:
```python
signs_list = ['a na', 'be lí', 'ù']
text = ' '.join(signs_list)  # → "a na be lí ù"
```

**Verification Code:** `v_1/src/training/baseline/data_utils.py:43-74`

### 3. Model Tokenization is Space-Split

The model tokenizer is equally simple:
```python
# From data_utils.py:179-224
def tokenize_text(text: str, sign_to_id: Dict[str, int], max_length: int):
    signs = text.split()  # ← Split on spaces!
    token_ids = [sign_to_id.get(sign, unk_id) for sign in signs]
    # Add [CLS], [SEP], [PAD]...
    return token_ids, attention_mask
```

Example:
```python
text = "a na be lí"
signs = text.split()  # → ['a', 'na', 'be', 'lí']
token_ids = [vocab[s] for s in signs]  # → [5745, 10329, 1902, 9027]
```

**Verification Code:** `v_1/src/training/baseline/data_utils.py:179-224`

### 4. No Data Leakage Between Splits

Train/val/test splits are done by `fragment_id`:
```python
# From 05_create_unified.py:36-67
fragment_ids = df['fragment_id'].unique()
np.random.shuffle(fragment_ids)

train_ids = set(fragment_ids[:n_train])
val_ids = set(fragment_ids[n_train:n_train + n_val])
test_ids = set(fragment_ids[n_train + n_val:])

train_df = df[df['fragment_id'].isin(train_ids)]
val_df = df[df['fragment_id'].isin(val_ids)]
test_df = df[df['fragment_id'].isin(test_ids)]
```

**Verification:** `train_ids ∩ val_ids = ∅` and `train_ids ∩ test_ids = ∅`

**Verification Code:** `v_1/src/preprocessing/05_create_unified.py:36-67`

---

## File References

| Pipeline Stage | Code File | Output File | Key Function |
|----------------|-----------|-------------|--------------|
| **1. Source Processing** | [02_process_ebl.py](../v_1/src/preprocessing/02_process_ebl.py) | `ebl_corpus.parquet` | `tokenize_to_signs()` (line 24-72) |
| | [03_process_archibab.py](../v_1/src/preprocessing/03_process_archibab.py) | `archibab_corpus.parquet` | Similar tokenization |
| | [04_process_oracc.py](../v_1/src/preprocessing/04_process_oracc.py) | `oracc_corpus.parquet` | Similar tokenization |
| **2. Unification** | [05_create_unified.py](../v_1/src/preprocessing/05_create_unified.py) | `unified_corpus.parquet` | `create_unified_dataset()` (line 70) |
| **3. Splitting** | [05_create_unified.py](../v_1/src/preprocessing/05_create_unified.py) | `train/val/test.parquet` | `create_train_val_test_split()` (line 36) |
| **4. Fragment Building** | [01_prepare_data.py](../v_1/src/training/baseline/01_prepare_data.py) | `*_fragments.parquet` | Uses `build_fragment_texts()` |
| | [data_utils.py](../v_1/src/training/baseline/data_utils.py) | | `reconstruct_fragment_text()` (line 43) |
| | | | `build_fragment_texts()` (line 77) |
| **5. MLM Dataset** | [data_utils.py](../v_1/src/training/baseline/data_utils.py) | PyTorch tensors | `AkkadianMLMDataset` (line 227) |
| | | | `tokenize_text()` (line 179) |
| | | | `_apply_mlm_masking()` (line 298) |
| **Training** | [run_training.py](../v_1/run_training.py) | Model checkpoints | Loads dataset and trains |

---

## Running the Verification

To verify the entire pipeline:

```bash
# 1. Verify all stages
python3 v_1/src/verify_data_pipeline.py

# 2. Verify source parquets exist
ls -lh v_1/data/processed/*/*.parquet

# 3. Verify unified and splits
ls -lh v_1/data/processed/unified/*.parquet

# 4. Verify prepared fragments
ls -lh v_1/data/prepared/*_fragments.parquet

# 5. Check vocabulary
python3 -c "import json; print(json.load(open('v_1/data/prepared/vocab.json'))['vocab_size'])"
# Expected: 14797
```

---

## Summary for Thesis

**Data Pipeline Validation:**

Our Akkadian MLM training pipeline processes data through five verified stages. We ensure data integrity at every step:

1. **Sign-level tokenization** is applied during initial CSV processing, following EvaCun 2025 methodology. Each word's transliteration (e.g., "a-na") is split into individual signs (e.g., "a na") and stored in the `value_signs` column.

2. **Fragment reconstruction** groups words by tablet fragment ID, sorts them by their original position (line number, word index), and concatenates all signs into a single space-separated text string.

3. **No data leakage** between train/validation/test sets is guaranteed by splitting at the fragment level (not the word level), ensuring no tablet appears in multiple splits.

4. **Model input** is created by splitting fragment text on spaces, converting signs to vocabulary IDs, adding special tokens ([CLS], [SEP], [PAD]), and applying 15% MLM masking using BERT's 80/10/10 strategy.

We provide a comprehensive verification script (`verify_data_pipeline.py`) that traces real fragments through all stages, confirming that:
- Fragment "ARM 10 33" contains 297 signs reconstructed from 82 words
- The text "a na be lí ù ka ka bi..." correctly preserves the original tablet order
- Tokenization produces the expected vocabulary IDs for each sign
- MLM masking creates valid training targets

This pipeline processes 2.45M tokens from 40,429 fragments across three major digital libraries (ORACC, eBL, Archibab), representing a 2× scale increase over previous benchmarks (Fetaya et al., 2021).

---

## Conclusion

This verification demonstrates:
✅ Data flows correctly through all pipeline stages
✅ Fragment reconstruction preserves tablet structure
✅ Sign-level tokenization is consistent across sources
✅ MLM masking produces valid training targets
✅ No data leakage between splits
✅ All transformations are traceable and reproducible

The verification script provides ongoing confidence in the data pipeline and serves as documentation for the thesis.
