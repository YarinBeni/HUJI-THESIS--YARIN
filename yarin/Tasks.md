### Akkadian Temporal Classification & MLM — Implementation Task List

**Updated**: January 28, 2026

**Goal**:
1. Train a baseline MLM model on unified Akkadian data (COMPLETE)
2. Create temporal classification baseline on Chunrong's 3-group corpus (NEXT)
3. Fine-tune MMBERT for comparison
4. Run SAE interpretability analysis

**Key constraints**
- **MLM Objective**: Masking — *not* causal LM
- **Original Data**: `v_1/data/processed/unified/{train,val,test}.parquet`
- **Evaluation Data**: `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet`
- **Analysis artifacts**: Save pre/post weights + embeddings + hidden states

---

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Design Decisions | COMPLETE |
| **Phase 1** | Baseline MLM Training | COMPLETE |
| **Phase 1.5** | Evaluation Corpora Preparation | COMPLETE |
| **Phase 2** | LLM Baseline Pipeline | **IN PROGRESS** |
| **Phase 3** | MMBERT Fine-tuning | PENDING |
| **Phase 4** | SAE Interpretability | PENDING |

---

### Phase 0 — Decisions to lock FINALIZED

- [x] **0.1 MLM masking granularity**
  - **Decision**: Mask **individual signs** (token-level).
  - Each sign from `value_signs` (split on spaces) is a separate token.
  - Example: `"a na be li"` -> tokens `["a", "na", "be", "li"]` -> mask individual signs.
  - **Justification**: See `yarin/justification/justification_sign_level_tokenization.md`

- [x] **0.2 Word-boundary representation**
  - **Decision**: **Option A** — No explicit boundary token.
  - Words are implicitly separated by the row structure in parquet.
  - When reconstructing text: join all `value_signs` from fragment rows with spaces.
  - Signs within a word are already space-separated in `value_signs`.

- [x] **0.3 Analysis layers schedule**
  - **Decision**: Every 4th layer from 0 to 16 (inclusive): **[0, 4, 8, 12, 16]**
  - For a 16-layer model, this gives 5 layers covering input embeddings through final layer.
  - Provides full "scan" of model depth without redundant adjacent layers.

**Deliverable**: Decisions documented above and in justification files.

---

### Phase 1 — Baseline PyTorch "torso" MLM COMPLETE

#### Model Architecture: "Simplified Aeneas Twin"
Based on the Aeneas paper (Assael et al., 2025). See `yarin/justification/justification_aeneas_twin_architecture.md`.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 384 | Embedding dimension |
| `d_ff` | 1,536 | MLP hidden dimension |
| `d_kv` | 32 | Per-head Q/K/V dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 16 | Transformer blocks |
| `vocab_size` | ~16,750 | ~16,740 signs + special tokens |
| `max_seq_len` | 768 | Maximum sequence length |
| Positional | **RoPE** | Rotary embeddings |
| Norm | Pre-Norm (RMSNorm) | T5-style |
| Head | 2-layer MLP | Restoration head |

#### 1A. Dataset view for training COMPLETE
- [x] **1A.1 Create a "text per fragment" builder**
  - **Implementation**: `v_1/src/training/baseline/data_utils.py::build_fragment_texts()`
  - Joins `value_signs` per fragment, sorted by (`line_num`, `word_idx`)
- [x] **1A.2 Build token vocabulary for signs**
  - **Implementation**: `v_1/src/training/baseline/data_utils.py::build_sign_vocabulary()`
  - Vocabulary: 14,797 tokens (5 special + 14,792 signs)
  - Saved to: `v_1/data/prepared/vocab.json`
- [x] **1A.3 Create PyTorch Dataset + Collator for MLM**
  - **Implementation**: `v_1/src/training/baseline/data_utils.py::AkkadianMLMDataset`
  - 15% masking, BERT-style 80/10/10 replacement
- [x] **1A.4 Fixed eval subset**
  - 500 deterministic fragments (seed=42)
  - Saved to: `v_1/data/prepared/eval_subset.parquet`

#### 1B. Model torso implementation (PyTorch) COMPLETE
- [x] **1B.1 Implement Simplified Aeneas Twin**
  - **Implementation**: `v_1/src/training/baseline/model.py::AeneasForMLM`
  - 16 layers, d_model=384, d_kv=32, RoPE, Pre-Norm
  - ~37M parameters
- [x] **1B.2 Add "return hidden states" support**
  - **Implementation**: `output_hidden_states=True, hidden_states_layers=[0,4,8,12,16]`
  - Returns Dict[layer_idx -> tensor]

#### 1C. Training + checkpointing (baseline) COMPLETE
- [x] **1C.1 Training script**
  - **Implementation**: `v_1/src/training/baseline/02_train.py`
  - Saves `baseline_init.pt`, `baseline_best.pt`, `baseline_last.pt`
  - AdamW optimizer with cosine annealing
- [x] **1C.2 Baseline embedding/hidden-state dumps (pre/post)**
  - Extracts embeddings and hidden states for layers [0, 4, 8, 12, 16]
  - Saves pre/post training artifacts

**Training Results**:
- 10 epochs, best val_loss = 3.02
- Training time: ~8.3 hours on Apple Silicon MPS
- Artifacts saved to `v_1/models/baseline/` (~4.3 GB)

---

### Phase 1.5 — Evaluation Corpora Preparation COMPLETE

#### 1.5A. Receive and analyze Chunrong's data COMPLETE
- [x] **1.5A.1 Receive normalized CSV files**
  - `archibab_nor.csv` (Old Babylonian - Group 1)
  - `oracc_let_adm_nor.csv` (Neo-Assyrian - Group 2)
  - `lbl_nor.csv` (Late Babylonian - Group 3)
  - Location: `v_1/data/processed/from_chungrong/`
  - Received: Jan 20-23, 2026

- [x] **1.5A.2 Analyze email thread and data structure**
  - Document: `yarin/emails_phase/EMAIL_ANALYSIS_AND_DATA_VERIFICATION.md`
  - Identified Nathan's 3 temporal-geographic groups
  - Verified file-to-group mapping

- [x] **1.5A.3 EDA on 3-groups corpus**
  - Notebook: `v_1/notebooks/03_eda_corpora.ipynb`
  - Domain analysis, provenance analysis, gap analysis
  - Domain merge recommendations implemented

#### 1.5B. Create unified evaluation corpus COMPLETE
- [x] **1.5B.1 Create unification script**
  - Script: `v_1/src/preprocessing/06_create_test_letters_copra.py`
  - Combines all 3 groups with metadata columns

- [x] **1.5B.2 Apply domain standardization**
  - Domain merges for Archibab (4 typo/variant fixes)
  - Filter ORACC for letters only (`domain == 'NALet'`)
  - Create `domain_standard` and `domain_finegrained` columns

- [x] **1.5B.3 Add metadata columns**
  - `temporal_group`: Group 1, Group 2, Group 3
  - `period`: Old Babylonian, Neo-Assyrian, Late Babylonian
  - `period_approx`: Human-readable date ranges
  - `corpus_source`: archibab, oracc, lbl

- [x] **1.5B.4 Save unified corpus**
  - Output: `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet`
  - CSV backup: `unified_3groups_akkadian_letters.csv`
  - Stats: 290,652 words, 6,577 texts

#### 1.5C. Metadata audit COMPLETE
- [x] **1.5C.1 Audit ORACC metadata**
  - Matched P-numbers to CDLI catalog
  - Recovered 96.2% period coverage
  - Document: `yarin/justification/task_2_5_metadata_audit_summary.md`

- [x] **1.5C.2 Create filtered evaluation subsets**
  - `corpus_a_archibab_2nd_mill.parquet` (2nd millennium)
  - `corpus_b_oracc_1st_mill.parquet` (1st millennium)

---

### Phase 2 — LLM Baseline Pipeline **IN PROGRESS**

**Goal**: Create LLM-based baseline predictions for temporal period classification on cleaned evaluation corpus (4,976 texts).

**Decision**: Implemented LLM-based approach using OpenRouter API with multiple models.

#### 2A. Evaluation corpus preparation COMPLETE
- [x] **2A.1 Prepare text-level dataset**
  - Script: `v_1/src/evaluation/01_prepare_texts.py`
  - Input: Word-level data from `unified_3groups_akkadian_letters.parquet`
  - Output: `texts_for_evaluation.parquet` (4,976 texts) + `.jsonl` format
  - Reconstructed full texts by joining words per fragment

- [x] **2A.2 Domain label cleanup**
  - Removed 29 texts with Unknown/nan domain labels (5,005 → 4,976)
  - Notebook: `v_1/notebooks/04_eda_evaluation.ipynb` (Section 13-14)
  - Justification: `yarin/justification/domain_label_cleanup.md`

- [x] **2A.3 Create token statistics**
  - Output: `texts_token_stats.json`
  - Average tokens per text: ~1,000-1,500 (estimate based on char count / 4)

#### 2B. LLM baseline implementation COMPLETE
- [x] **2B.1 OpenRouter API integration**
  - Script: `v_1/src/evaluation/02_llm_baseline.py`
  - Uses `requests` library (no openai package needed)
  - Environment variable: `OPENROUTER_API_KEY`
  - Direct HTTPS POST to `https://openrouter.ai/api/v1/chat/completions`

- [x] **2B.2 Model registry**
  - Config: `v_1/src/evaluation/config.py`
  - **Free models** (Phase A): GPT-OSS-20B, GPT-OSS-120B, Gemini-2.5-Pro, Mistral-Small, DeepSeek-v3, DeepSeek-R1
  - **Open-source models** (Phase C): Qwen family, Mixtral, Mistral, DeepSeek
  - **Paid models** (Phase D): Gemini-2.0-Flash, GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet, Grok-2

- [x] **2B.3 Prompt engineering**
  - **Format**: Markdown fields with example (more robust than JSON)
  - **Structure**: `**Period**: ...`, `**Century**: ...`, `**Domain**: ...`, `**Place**: ...`, `**Confidence**: ...`, `**Reasoning**: ...`
  - **Max tokens**: 4096 (allows reasoning models to think before answering)
  - **Example-driven**: Includes sample response in prompt

- [x] **2B.4 Response parsing**
  - Primary: Regex extraction of markdown fields
  - Fallback: JSON parsing for models that ignore format
  - Handles reasoning models: extracts from `reasoning_details` when `content` is empty

- [x] **2B.5 Caching & resume support**
  - Per-model JSONL cache in `v_1/data/evaluation_corpora/cache/`
  - Automatically resumes from failures
  - Deduplicates by fragment_id

- [x] **2B.6 Rate limiting & retries**
  - Configurable sleep between API calls (default: 0.5s)
  - Retry logic: 3 attempts with exponential backoff
  - Timeout: 60 seconds per request

#### 2C. Results aggregation & evaluation COMPLETE
- [x] **2C.1 Aggregation pipeline**
  - Script: `v_1/src/evaluation/03_aggregate_results.py`
  - Loads all model caches, pivots to wide format
  - Merges with ground truth labels
  - Output: `baseline_predictions.parquet`

- [x] **2C.2 Evaluation metrics**
  - Script: `v_1/src/evaluation/04_evaluate_baseline.py`
  - Metrics: Accuracy, F1 (macro/weighted), Precision, Recall
  - Confusion matrices per model
  - Per-period and per-group breakdowns
  - Token usage statistics

- [x] **2C.3 Evaluation outputs**
  - Markdown report: `baseline_results_report.md`
  - JSON metrics: `baseline_metrics.json`
  - Includes confidence distribution, token costs

#### 2D. Testing & validation COMPLETE
- [x] **2D.1 Initial test run**
  - 10 Old Babylonian texts with GPT-OSS-20B free model
  - Parse rate: 90% (9/10)
  - Period accuracy: 22.2% (2/9 correct)
  - Model bias: 6/9 misclassified as Neo-Assyrian
  - Token usage: 24,489 total (9,767 input + 14,722 output)
  - Speed: ~14 seconds per text on free tier

#### 2E. Next steps IN PROGRESS
- [ ] **2E.1 Full corpus evaluation**
  - Run all 4,976 texts with GPT-OSS-20B
  - Estimated time: ~20 hours on free tier
  - Estimated tokens: ~10M total

- [ ] **2E.2 Multi-model comparison**
  - Test additional free models: Gemini, DeepSeek, Mistral
  - Compare accuracy across models
  - Document which models work best

- [ ] **2E.3 Prompt refinement**
  - Analyze failure patterns
  - Refine prompt to improve accuracy
  - Consider few-shot examples

**Deliverables COMPLETED**:
- ✅ Complete LLM baseline pipeline (4 scripts + config)
- ✅ OpenRouter integration with model registry
- ✅ Caching & resume support
- ✅ Evaluation pipeline with metrics
- ✅ Initial test results (10 texts)

**Deliverables PENDING**:
- ⏳ Full corpus evaluation results (4,976 texts)
- ⏳ Multi-model comparison
- ⏳ Final baseline performance report

---

### Phase 3 — MMBERT Fine-tuning (Hugging Face, MLM)

**Goal**: Fine-tune MMBERT for comparison with from-scratch baseline.

#### 3A. Sanity-check tokenizer coverage
- [ ] **3A.1 Tokenization audit**
  - Run MMBERT tokenizer over a sample of `text_signs`
  - Report: % `[UNK]` (if applicable), average pieces per sign/word, max length stats
  - Decide if preprocessing tweaks are needed (spacing, boundary token, normalization)

**Deliverable**: small report (markdown or txt) with tokenization stats.

#### 3B. Fine-tuning pipeline
- [ ] **3B.1 Build HF `datasets` object for train/val/test**
  - One row per fragment (or chunked sequences)
- [ ] **3B.2 Implement MLM data collator**
  - Use HF collator or custom masking to match your baseline
- [ ] **3B.3 Save "pre-finetune" artifacts**
  - Save MMBERT model weights before updates
  - Dump embeddings + hidden states on the same fixed eval subset (same layers schedule)
- [ ] **3B.4 Fine-tune**
  - Train with HF Trainer/Accelerate
  - Save checkpoints and final model
- [ ] **3B.5 Save "post-finetune" artifacts**
  - Save weights after fine-tune
  - Dump embeddings + hidden states again (same eval subset, same layer indices)

**Deliverables**
- `mmbert_pre_ft_state_dict.pt`, `mmbert_post_ft_state_dict.pt`
- `mmbert_pre_hidden_states_layers_*.pt`, `mmbert_post_hidden_states_layers_*.pt`
- HF output dir with config/tokenizer (if applicable)

#### 3C. Compare MMBERT vs Baseline for temporal classification
- [ ] **3C.1 Run same classification pipeline on MMBERT**
- [ ] **3C.2 Compare results with baseline**
- [ ] **3C.3 Document findings**

---

### Phase 4 — SAE interpretability (baseline + MMBERT)

#### 4A. Standardize activation dataset
- [ ] **4A.1 Choose activation source**
  - Use hidden states you saved in Phase 1C/3B, or re-extract on demand
- [ ] **4A.2 Create a consistent on-disk schema**
  - Include: model id, layer id, token ids, attention mask, mapping back to `fragment_id`

#### 4B. SAE training
- [ ] **4B.1 Train SAE per chosen layer (baseline torso)**
  - Start with a middle-ish layer + a late layer
  - Track sparsity, dead features, reconstruction loss
- [ ] **4B.2 Train SAE per chosen layer (MMBERT)**
  - Same procedure for comparability

#### 4C. Feature analysis
- [ ] **4C.1 Top-activating examples per feature**
- [ ] **4C.2 Feature -> linguistic probe notebooks/scripts**
  - e.g., suffixes, determinatives, genre markers, temporal period markers

**Deliverables**
- SAE weights + metrics json per (model, layer)
- Lightweight feature inspection utilities (histograms, top-k contexts)

---

## Data Locations Summary

| Data | Location | Description |
|------|----------|-------------|
| Original unified dataset | `v_1/data/processed/unified/` | Train/val/test parquets (~2.45M words) |
| Chunrong's raw files | `v_1/data/processed/from_chungrong/` | 3 CSV files |
| Unified 3-groups corpus | `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet` | Word-level (290,652 words, 6,577 texts) |
| **Evaluation texts** | `v_1/data/evaluation_corpora/texts_for_evaluation.parquet` | **Text-level (4,976 texts) - Use for LLM eval** |
| LLM prediction caches | `v_1/data/evaluation_corpora/cache/*.jsonl` | Per-model JSONL caches |
| Aggregated predictions | `v_1/data/evaluation_corpora/baseline_predictions.parquet` | All models + ground truth |
| Evaluation reports | `v_1/data/evaluation_corpora/baseline_results_report.md` | Metrics, confusion matrices |
| Trained baseline model | `v_1/models/baseline/baseline_best.pt` | Best checkpoint |
| Pre/post embeddings | `v_1/models/baseline/baseline_{pre,post}_embeddings.pt` | For analysis |
| Hidden states | `v_1/models/baseline/baseline_{pre,post}_hidden_states_layer_*.pt` | 5 layers |

---

## Quick Reference Commands

```bash
# Run baseline training (Phase 1 - already complete)
python3 v_1/run_training.py --epochs 10

# Create unified 3-groups corpus (Phase 1.5 - already complete)
python3 v_1/src/preprocessing/06_create_test_letters_copra.py

# Prepare text-level data for LLM evaluation (Phase 2 - already complete)
python3 v_1/src/evaluation/01_prepare_texts.py

# Set OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Run LLM baseline predictions (Phase 2)
# Test run (10 samples)
python3 v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --sample 10

# Full run (4,976 texts)
python3 v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b

# Aggregate results and evaluate
python3 v_1/src/evaluation/03_aggregate_results.py
python3 v_1/src/evaluation/04_evaluate_baseline.py

# List available models
python3 v_1/src/evaluation/02_llm_baseline.py --list-models

# Load evaluation corpus
python3 -c "
import pandas as pd
df = pd.read_parquet('v_1/data/evaluation_corpora/texts_for_evaluation.parquet')
print(f'Loaded {len(df):,} texts')
print(df.groupby('period').size())
"
```

---

## References

- **Baseline torso + SAE pipeline (older)**: `v_0/RESTORATION_PROJECT_PLAN.md`
- **Torso model implementation**: `v_1/src/training/baseline/model.py`
- **Torso training**: `v_1/src/training/baseline/02_train.py`
- **SAE utilities (older)**: `v_0/src/06_sae_memory_optimized.py`, `v_0/src/07_inspect_sae.py`
- **Email analysis**: `yarin/emails_phase/EMAIL_ANALYSIS_AND_DATA_VERIFICATION.md`
- **Metadata audit**: `yarin/justification/task_2_5_metadata_audit_summary.md`
