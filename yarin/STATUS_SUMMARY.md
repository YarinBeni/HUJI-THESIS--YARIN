# Akkadian LLM Project - Status Summary

**Date**: January 28, 2026
**Status**: PHASE 2 IN PROGRESS - LLM Baseline Pipeline Complete
**Current Focus**: Testing and refining LLM baseline evaluation pipeline
**Next Phase**: Full corpus evaluation + MMBERT fine-tuning

---

## Timeline Overview

| Phase | Status | Date |
|-------|--------|------|
| Phase 0: Design Decisions | COMPLETE | Dec 2025 |
| Phase 1: Baseline MLM Training | COMPLETE | Dec 29, 2025 |
| Phase 1.5: Evaluation Corpora | COMPLETE | Jan 25, 2026 |
| **Phase 2: LLM Baseline Pipeline** | **IN PROGRESS** | **Jan 28, 2026** |
| Phase 3: MMBERT Fine-tuning | PENDING | - |
| Phase 4: SAE Interpretability | PENDING | - |

---

## What's New (January 2026)

### New Data from Chunrong (Jan 20-23, 2026)

Received 3 normalized CSV files for **temporal classification** task:

| File | Period | Group | Words | Texts |
|------|--------|-------|-------|-------|
| `archibab_nor.csv` | Old Babylonian (~1800 BCE) | Group 1 | 80,059 | 1,526 |
| `oracc_let_adm_nor.csv` | Neo-Assyrian (9-7 cent BCE) | Group 2 | 229,191 | 4,007 |
| `lbl_nor.csv` | Late Babylonian (~600 BCE) | Group 3 | 74,169 | 1,044 |

**Location**: `v_1/data/processed/from_chungrong/`

### Unified 3-Groups Corpus (Jan 25, 2026)

Created unified evaluation corpus combining all 3 temporal groups:

| Metric | Value |
|--------|-------|
| **Total Words** | 290,652 |
| **Total Texts** | 6,577 |
| **File** | `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet` |

**Script**: `v_1/src/preprocessing/06_create_test_letters_copra.py`

### Metadata Audit (Jan 2026)

- CDLI matching recovered 96.2% period coverage for ORACC
- Created filtered evaluation corpora by period
- See `yarin/justification/task_2_5_metadata_audit_summary.md`

### Data Cleanup (Jan 28, 2026)

**Domain Label Cleanup**: Removed 29 texts with Unknown/nan domain labels from evaluation corpus
- Fixed string "nan" values (22 texts)
- Removed "Other" domain labels (22 texts)
- Removed "Unknown" domain labels (7 texts)
- **Final corpus**: 4,976 texts (down from 5,005)
- **Justification**: `yarin/justification/domain_label_cleanup.md`
- **Corpus size reasoning**: `yarin/justification/evaluation_corpus_size_5005_texts.md`

### LLM Baseline Pipeline (Jan 28, 2026)

Created complete end-to-end LLM baseline evaluation pipeline for temporal classification:

**Pipeline Components:**
1. **Text Preparation** (`01_prepare_texts.py`): Reconstructs fragment-level texts from word-level data
2. **LLM Predictions** (`02_llm_baseline.py`): Calls OpenRouter API with multiple model support
3. **Results Aggregation** (`03_aggregate_results.py`): Combines predictions from all models
4. **Evaluation Metrics** (`04_evaluate_baseline.py`): Computes accuracy, F1, confusion matrices

**Key Features:**
- **OpenRouter Integration**: Uses requests library for API calls (no openai package needed)
- **Model Registry**: Free models (GPT-OSS-20B, Gemini, DeepSeek, etc.), open-source, and paid tiers
- **Reasoning Model Support**: Handles GPT-OSS-20B's reasoning tokens (extracts from `reasoning_details`)
- **Markdown Prompt Format**: Structured field format with example (more robust than JSON)
- **Caching & Resume**: JSONL cache per model, automatically resumes from failures
- **Rate Limiting**: Configurable sleep between API calls
- **Token Tracking**: Full usage statistics per model

**Prompt Engineering:**
- Markdown field format: `**Period**: ...`, `**Domain**: ...`, `**Place**: ...`
- Example-driven (includes sample response in prompt)
- 4096 max tokens (allows reasoning models to think before answering)
- Fallback JSON parsing for models that ignore format

**Initial Test Results (10 texts, GPT-OSS-20B free model):**
- Parse success: 9/10 (90%)
- Period accuracy: 22.2% (2/9 correct)
- Token usage: 9,767 input + 14,722 output = 24,489 total
- Speed: ~14 seconds per text on free tier

**Location**: `v_1/src/evaluation/`

---

## Completed Work

### Phase 1: Baseline MLM Training (Dec 2025)

#### 1. Data Acquisition & Processing
*   **ORACC**: Downloaded and processed (~1.4M tokens).
*   **eBL**: Processed (~1M tokens).
*   **Archibab**: Processed (~65k tokens).
*   **Unified Dataset**: Merged all sources into `v_1/data/processed/unified/`.
*   **Train/Val/Test Splits**: Created and verified (80/10/10 by fragment_id, NO LEAKAGE).

#### 2. Model Training
*   **Architecture**: Simplified Aeneas Twin (37M params)
*   **Training**: 10 epochs, best val_loss = 3.02
*   **Artifacts**: Pre/post embeddings and hidden states saved (~4.3 GB)

#### 3. Original Unified Dataset Stats

| Metric | Value |
|--------|-------|
| **Total Words** | 2,450,094 |
| **Total Signs** | 4,894,744 |
| **Unique Signs** | 16,740 |
| **Total Texts** | 40,429 |

| Source | Tokens | % |
|--------|--------|---|
| **ORACC** | 1,385,932 | 56.6% |
| **eBL** | 998,353 | 40.7% |
| **Archibab** | 65,809 | 2.7% |

---

### Phase 1.5: Evaluation Corpora (Jan 2026)

#### 1. Chunrong Data Analysis
- Email thread analysis documented in `yarin/emails_phase/EMAIL_ANALYSIS_AND_DATA_VERIFICATION.md`
- 3 temporal groups identified per Nathan's geographic-temporal framework
- EDA completed in `v_1/notebooks/03_eda_corpora.ipynb`

#### 2. Unified Corpus Creation
- Combined all 3 groups with proper metadata columns
- Added: `temporal_group`, `period`, `period_approx`, `corpus_source`
- Domain standardization applied (fine-grained and standard labels)

#### 3. Key Decisions
| Decision | Choice | Notes |
|----------|--------|-------|
| Include Group 2 (Neo-Assyrian)? | YES | Originally planned to skip, now including all 3 |
| Domain filtering | Letters only for ORACC | Filter `domain == 'NALet'` |
| Domain merges for Archibab | 4 merges applied | Fix typos, standardize labels |

---

### Phase 2: LLM Baseline Pipeline (Jan 28, 2026)

#### 2.1 Text Preparation
- Reconstructed fragment-level texts from word-level rows
- Output: `texts_for_evaluation.parquet` (4,976 texts) and `.jsonl` format
- Token statistics: avg tokens per text, prompt template size

#### 2.2 OpenRouter API Integration
- Direct `requests` library implementation (no openai package dependency)
- Model registry with 3 tiers: Free (GPT-OSS, Gemini, DeepSeek), Open-source, Paid
- API key support via environment variable: `OPENROUTER_API_KEY`

#### 2.3 Prompt Engineering
- **Format**: Markdown fields with structured example
- **Max tokens**: 4096 (supports reasoning models like GPT-OSS-20B)
- **Fields**: Period, Century, Domain, Place, Confidence, Reasoning
- **Parser**: Regex-based markdown extraction + JSON fallback

#### 2.4 Key Technical Fixes
| Issue | Solution |
|-------|----------|
| Parse errors (empty responses) | Reasoning models spend tokens thinking before answering. Increased max_tokens from 300 to 4096 |
| Content in `reasoning_details` | Extract from `message['reasoning_details']` when `content` is empty |
| Hit 4096 token ceiling | Some texts still fail (~10%), marked as "Parse Error" in evaluation |

#### 2.5 Evaluation Pipeline
- **Aggregation**: Pivots predictions to wide format (one row per text, columns per model)
- **Metrics**: Accuracy, F1 (macro/weighted), Precision, Recall, Confusion Matrix
- **Breakdowns**: Per-period, per-group, per-model comparisons
- **Outputs**: Markdown report + JSON metrics

#### 2.6 Initial Test Results
**Test run: 10 Old Babylonian texts, GPT-OSS-20B free model**
- Parse rate: 90% (9/10)
- Period accuracy: 22.2% (2/9 correct)
- Domain accuracy: 0%
- Confusion: Model biased toward Neo-Assyrian (6/9 misclassified)
- Token usage: 24,489 total (9,767 input + 14,722 output)

#### 2.7 Technical Challenges & Solutions

| Challenge | Root Cause | Solution |
|-----------|-----------|----------|
| Parse errors (40% initially) | Hit token ceiling, responses truncated | Increased `MAX_COMPLETION_TOKENS` 300 → 4096. Parse rate improved to 90% |
| Empty `content` field | GPT-OSS-20B is a reasoning model — puts answer in `reasoning_details` | Extract from `message['reasoning_details']` when `content` is empty |
| High token usage | Reasoning models use 1500-2500 tokens thinking + 200 for answer | Accept overhead; 4096 max allows most texts to complete (~10% still fail) |

#### 2.8 Sample Predictions

**Text 1 (ARM 10 33)** — WRONG: Predicted Neo-Assyrian (ground truth: Old Babylonian). Model cited "Neo-Assyrian epistolary formulas" incorrectly.

**Text 3 (ARM 10 5)** — CORRECT: Predicted Old Babylonian, 18th century BCE (exact match). Model identified "Old Babylonian grammatical features."

**Text 10 (ARM 27 160)** — PARSE ERROR: Used all 4096 tokens on reasoning, never produced formatted answer.

#### 2.9 Usage Guide

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Test run (10 samples)
python3 v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --sample 10

# Full run (4,976 texts, ~20 hours free tier)
python3 v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b

# Dry run (estimate costs)
python3 v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --dry-run

# Aggregate + evaluate
python3 v_1/src/evaluation/03_aggregate_results.py
python3 v_1/src/evaluation/04_evaluate_baseline.py
```

**Model selection & budget**: See `yarin/justification/model_selection_phase2.md`

---

## Training Results (Phase 1)

| Metric | Value |
|--------|-------|
| **Epochs** | 10 |
| **Best Val Loss** | 3.0204 |
| **Final Train Loss** | 2.6506 |
| **Training Time** | ~8.3 hours (50 min/epoch) |
| **Device** | Apple Silicon MPS |
| **Batch Size** | 8 (auto-optimized) |
| **Parameters** | 36,705,229 |

### Loss Progression
```
Epoch  1: train=4.9800, val=4.5458
Epoch  2: train=4.0891, val=4.0711
Epoch  3: train=3.6677, val=3.8166
Epoch  4: train=3.3943, val=3.6407
Epoch  5: train=3.1580, val=3.4887
Epoch  6: train=2.9819, val=3.2902
Epoch  7: train=2.8562, val=3.2377
Epoch  8: train=2.7435, val=3.1103
Epoch  9: train=2.6861, val=3.1124
Epoch 10: train=2.6506, val=3.0204  <- Best model saved
```

---

## Output Artifacts

### Phase 1 Artifacts (v_1/models/baseline/)

| File | Size | Description |
|------|------|-------------|
| `baseline_init.pt` | 140 MB | Initial random weights |
| `baseline_best.pt` | 420 MB | Best checkpoint (epoch 10) |
| `baseline_last.pt` | 420 MB | Final checkpoint |
| `baseline_pre_embeddings.pt` | 22 MB | Embeddings before training |
| `baseline_post_embeddings.pt` | 22 MB | Embeddings after training |
| `baseline_pre_hidden_states_layer_*.pt` | 375 MB each | Hidden states before (5 files) |
| `baseline_post_hidden_states_layer_*.pt` | 375 MB each | Hidden states after (5 files) |
| `training_stats.json` | <1 KB | Training metrics |

**Total: ~4.3 GB of artifacts saved**

### Phase 1.5 Artifacts (v_1/data/evaluation_corpora/)

| File | Size | Description |
|------|------|-------------|
| `unified_3groups_akkadian_letters.parquet` | 2.8 MB | Combined 3-groups corpus |
| `unified_3groups_akkadian_letters.csv` | 49 MB | CSV version |
| `corpus_a_archibab_2nd_mill.parquet` | 0.7 MB | 2nd millennium subset |
| `corpus_b_oracc_1st_mill.parquet` | 2.7 MB | 1st millennium subset |

### Phase 2 Artifacts (v_1/data/evaluation_corpora/ + v_1/src/evaluation/)

| File | Size | Description |
|------|------|-------------|
| `texts_for_evaluation.parquet` | 1.3 MB | Text-level data with ground truth (4,976 texts) |
| `texts_for_evaluation.jsonl` | - | JSONL version for API processing |
| `texts_token_stats.json` | <1 KB | Token statistics |
| `cache/gpt-oss-20b.jsonl` | ~10 KB | Cached predictions (10 texts test) |
| `baseline_predictions.parquet` | - | Aggregated predictions with ground truth |
| `baseline_results_report.md` | - | Evaluation report (accuracy, F1, confusion matrices) |
| `baseline_metrics.json` | - | Machine-readable metrics |

**Scripts:**
- `v_1/src/evaluation/01_prepare_texts.py` - Text reconstruction
- `v_1/src/evaluation/02_llm_baseline.py` - LLM predictions via OpenRouter
- `v_1/src/evaluation/03_aggregate_results.py` - Results aggregation
- `v_1/src/evaluation/04_evaluate_baseline.py` - Metrics computation
- `v_1/src/evaluation/config.py` - Model registry & prompt template

---

## Next Steps: Phase 2 Completion + Phase 3

### Immediate Priority
1. **Complete LLM baseline evaluation** (Phase 2)
   - Run full corpus evaluation (4,976 texts) with GPT-OSS-20B
   - Test additional free models (Gemini, DeepSeek, Mistral)
   - Refine prompt for better accuracy
   - Document final baseline performance

2. **MMBERT fine-tuning** (Phase 3)
   - Fine-tune MMBERT on same 4,976-text corpus
   - Compare against LLM baseline and from-scratch baseline
   - Evaluate temporal period classification accuracy

3. **SAE interpretability** (Phase 4)
   - Extract features from trained models
   - Analyze temporal period markers

### Full Task List
See `yarin/Tasks.md` for detailed task breakdown.

---

## Documentation Index

| File | Purpose | Updated |
|------|---------|---------|
| `yarin/STATUS_SUMMARY.md` | This file - current status | Feb 22, 2026 |
| `yarin/Tasks.md` | Full implementation checklist | Jan 28, 2026 |
| `yarin/PROGRESS.md` | Detailed research notes | Dec 28, 2025 |
| `yarin/MENTOR_UPDATE_PHASE1.md` | Phase 1 mentor update | Dec 29, 2025 |
| `yarin/emails_phase/` | Chunrong data analysis (local only, not in git) | Jan 23, 2026 |
| `yarin/justification/model_selection_phase2.md` | LLM model selection & budget | Jan 28, 2026 |
| `yarin/justification/task_2_5_metadata_audit_summary.md` | Metadata audit results | Jan 2026 |
| `yarin/justification/evaluation_corpus_size_5005_texts.md` | Why 5,005 texts not 6,577 | Jan 28, 2026 |
| `yarin/justification/domain_label_cleanup.md` | Domain label cleanup (5,005→4,976) | Jan 28, 2026 |
| `yarin/justification/justification_mlm.md` | Why MLM over causal LM | Dec 2025 |
| `yarin/justification/justification_sign_level_tokenization.md` | Why sign-level tokens | Dec 2025 |
| `yarin/justification/justification_aeneas_twin_architecture.md` | Model architecture | Dec 2025 |
| `yarin/justification/data_source_summary.md` | Dataset composition | Dec 2025 |

---

## Repository Structure

```
v_1/
├── src/
│   ├── preprocessing/           # Data download & processing
│   │   ├── 01_download_oracc.py
│   │   ├── 02_process_ebl.py
│   │   ├── 03_process_archibab.py
│   │   ├── 04_process_oracc.py
│   │   ├── 05_create_unified.py
│   │   └── 06_create_test_letters_copra.py
│   ├── evaluation/              # NEW - LLM baseline pipeline
│   │   ├── config.py            # Model registry & prompt
│   │   ├── 01_prepare_texts.py  # Text reconstruction
│   │   ├── 02_llm_baseline.py   # LLM predictions (OpenRouter)
│   │   ├── 03_aggregate_results.py  # Results aggregation
│   │   └── 04_evaluate_baseline.py  # Metrics computation
│   ├── training/
│   │   ├── baseline/            # Aeneas Twin training
│   │   │   ├── data_utils.py
│   │   │   ├── model.py
│   │   │   ├── 01_prepare_data.py
│   │   │   └── 02_train.py
│   │   └── mmbert/              # (Phase 3)
│   └── analysis/                # Metadata analysis scripts
│       ├── corpus_diagnostic.py
│       ├── cdli_period_matcher.py
│       └── cdli_join_diagnostic.py
├── data/
│   ├── raw/                     # Original data
│   ├── processed/
│   │   ├── unified/             # Original unified dataset
│   │   └── from_chungrong/      # Chunrong's 3 CSV files
│   ├── evaluation_corpora/      # Evaluation data + LLM results
│   │   ├── unified_3groups_akkadian_letters.parquet
│   │   ├── texts_for_evaluation.parquet  # Text-level (4,976 texts)
│   │   ├── cache/               # Model prediction caches (JSONL)
│   │   ├── baseline_predictions.parquet  # Aggregated results
│   │   ├── baseline_results_report.md    # Evaluation report
│   │   └── baseline_metrics.json         # Metrics
│   └── prepared/                # Training-ready data
├── models/
│   ├── baseline/                # Trained baseline checkpoints
│   └── mmbert/                  # (Phase 3)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_unified_dataset_eda.ipynb
│   ├── 03_eda_corpora.ipynb     # 3-groups EDA
│   └── 04_eda_evaluation.ipynb  # NEW - Evaluation corpus EDA + cleanup
└── run_training.py              # Training launcher
```

---

## Model Architecture: Simplified Aeneas Twin

| Parameter | Value |
|-----------|-------|
| `d_model` | 384 |
| `d_ff` | 1,536 |
| `d_kv` | 32 (per head) |
| `num_heads` | 8 |
| `num_layers` | 16 |
| `vocab_size` | 14,797 |
| `max_seq_len` | 768 |
| Positional | RoPE |
| Norm | Pre-Norm (RMSNorm) |
| Head | 2-layer MLP |
| **Total Params** | **36,705,229** |
