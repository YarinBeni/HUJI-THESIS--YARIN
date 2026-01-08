# Akkadian LLM Project - Status Summary

**Date**: December 29, 2025
**Status**: PHASE 1 TRAINING COMPLETE
**Next Phase**: Phase 2 - MMBERT fine-tuning

---

## ✅ What's Been Completed

### 1. Data Acquisition & Processing
*   **ORACC**: Downloaded and processed (~1.4M tokens).
*   **eBL**: Processed (~1M tokens).
*   **Archibab**: Processed (~65k tokens).
*   **Unified Dataset**: Merged all sources into `v_1/data/processed/unified/`.
*   **Train/Val/Test Splits**: Created and verified (80/10/10 by fragment_id, NO LEAKAGE).

### 2. Exploratory Data Analysis (EDA)
**Notebook**: `v_1/notebooks/02_unified_dataset_eda.ipynb`
*   Analyzed source distribution, vocabulary sizes, and tokenization.
*   Confirmed data quality and split integrity (no leakage).
*   **Summary Document**: `yarin/justification/data_source_summary.md`

### 3. Critical Decisions Finalized ✅
| Decision | Choice | Justification |
|----------|--------|---------------|
| **Objective** | MLM (Masked LM) | `justification_mlm.md` |
| **Tokenization** | Sign-level (split `value_signs` on spaces) | `justification_sign_level_tokenization.md` |
| **Word Boundaries** | Implicit (no explicit marker) | Row structure preserves word info |
| **Masking** | 15%, BERT 80/10/10 strategy | Standard MLM practice |
| **SAE Layers** | Every 4th layer: [0, 4, 8, 12, 16] | Full depth coverage |
| **Model Architecture** | "Simplified Aeneas Twin" (16L, d=384, RoPE) | `justification_aeneas_twin_architecture.md` |

---

## 📂 Documentation & Justification

| File | Purpose |
|------|---------|
| `yarin/justification/justification_mlm.md` | Why MLM over causal LM |
| `yarin/justification/justification_sign_level_tokenization.md` | Why sign-level tokens |
| `yarin/justification/justification_aeneas_twin_architecture.md` | Model architecture details |
| `yarin/justification/data_source_summary.md` | Dataset composition & stats |
| `yarin/justification/VALIDATION_PHASE1_TRAINING.md` | **NEW** - Training validation with proofs |
| `yarin/Tasks.md` | Full implementation task list |

---

## 📊 Unified Dataset Stats

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

| Split | Words | Texts |
|-------|-------|-------|
| **Train** | 1,960,636 | 32,343 |
| **Val** | 253,798 | 4,042 |
| **Test** | 235,660 | 4,044 |

---

## ✅ Phase 1 COMPLETE - Baseline MLM Training

### Phase 1A: Dataset Pipeline ✅
- [x] **1A.1** Fragment text builder → `v_1/src/training/baseline/data_utils.py`
- [x] **1A.2** Sign vocabulary → 14,797 tokens in `v_1/data/prepared/vocab.json`
- [x] **1A.3** PyTorch MLM Dataset → `AkkadianMLMDataset` class
- [x] **1A.4** Fixed eval subset → 500 fragments in `v_1/data/prepared/eval_subset.parquet`

### Phase 1B: Model Implementation ✅
- [x] **1B.1** Simplified Aeneas Twin → `v_1/src/training/baseline/model.py` (~37M params)
- [x] **1B.2** Hidden states extraction → layers [0, 4, 8, 12, 16]

### Phase 1C: Training Script ✅
- [x] **1C.1** Training script → `v_1/run_training.py` (launcher)
- [x] **1C.2** Pre/post embedding extraction → Built into training script

### Phase 1D: Training Execution ✅ **COMPLETE**
- [x] **1D.1** 10 epochs of MLM training completed
- [x] **1D.2** Pre/post hidden states saved for layers [0, 4, 8, 12, 16]
- [x] **1D.3** Pre/post embeddings saved

---

## 📈 Training Results

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
Epoch 10: train=2.6506, val=3.0204  ← Best model saved
```

---

## 📦 Output Artifacts (v_1/models/baseline/)

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

---

## 🚀 NEXT: Phase 2 - MMBERT Fine-tuning

See `yarin/Tasks.md` for detailed Phase 2 tasks:
- [ ] **2A.1** Tokenization audit (run MMBERT tokenizer on our data)
- [ ] **2B.1** Build HF datasets object
- [ ] **2B.2** Implement MLM data collator
- [ ] **2B.3** Save pre-finetune artifacts
- [ ] **2B.4** Fine-tune MMBERT
- [ ] **2B.5** Save post-finetune artifacts

---

## 📁 Repository Structure

```
v_1/
├── src/
│   ├── preprocessing/           # Data download & processing
│   │   ├── 01_download_oracc.py
│   │   ├── 02_process_ebl.py
│   │   ├── 03_process_archibab.py
│   │   ├── 04_process_oracc.py
│   │   └── 05_create_unified.py
│   ├── training/
│   │   ├── baseline/            # Aeneas Twin training ✅
│   │   │   ├── data_utils.py
│   │   │   ├── model.py
│   │   │   ├── 01_prepare_data.py
│   │   │   └── 02_train.py
│   │   └── mmbert/              # (Phase 2)
│   └── analysis/                # (Phase 3 - SAE)
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Processed parquets
│   └── prepared/                # Training-ready data ✅
├── models/
│   ├── baseline/                # Trained baseline checkpoints ✅
│   └── mmbert/                  # (Phase 2)
├── notebooks/
└── run_training.py              # Convenient training launcher ✅
```

---

## 🏗️ Model Architecture: Simplified Aeneas Twin

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
