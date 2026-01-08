# Research Update: Phase 1 Complete - Baseline MLM Training

**To:** Thesis Mentor
**From:** Yarin
**Date:** December 29, 2025
**Status:** Phase 1 Complete, Ready for Phase 2

---

## Executive Summary

I've successfully completed **Phase 1** of my Akkadian language modeling project: building and training a baseline Masked Language Model (MLM) from scratch using a unified dataset of ~2.5M Akkadian words (4.9M individual signs). This represents **more than 2x the data** used in previous state-of-the-art work (Fetaya et al., 2021).

**Key achievements:**
- ✅ Unified dataset from 3 sources (ORACC, eBL, Archibab)
- ✅ Custom PyTorch model implemented ("Simplified Aeneas Twin", 37M params)
- ✅ 10 epochs of training completed (best val_loss: 3.02)
- ✅ Pre/post training artifacts saved for interpretability analysis

**Next steps:** Fine-tune MMBERT for comparison (Phase 2), then run SAE interpretability analysis (Phase 3)

---

## 1. Research Motivation & Design Decisions

### 1.1 Why MLM Instead of Causal LM?

**Decision:** Use Masked Language Modeling (BERT-style) rather than next-token prediction (GPT-style)

**Justification:** (See `yarin/justification/justification_mlm.md`)

1. **Task Alignment:** The restoration task for damaged tablets is inherently bidirectional - scholars use context from *both sides* of a gap
2. **Architectural Precedent:** MMBERT (state-of-the-art multilingual encoder) and ModernBERT demonstrate that encoder-only MLM models outperform larger decoder-only models on understanding tasks
3. **Empirical Evidence:** Fetaya et al. (2021) showed that zero-shot multilingual BERT *never trained on Akkadian* outperformed monolingual Akkadian models, demonstrating MLM's power in low-resource settings

**Implementation:** 15% masking with BERT 80/10/10 strategy (80% replaced with [MASK], 10% random, 10% unchanged)

---

### 1.2 Why Sign-Level Tokenization?

**Decision:** Tokenize at the **sign level** by splitting `value_signs` on spaces, NOT at the word level

**Justification:** (See `yarin/justification/justification_sign_level_tokenization.md`)

1. **SOTA Precedent:** EvaCun 2025 Shared Task explicitly uses "transliterated signs as minimal units"
2. **Vocabulary Efficiency:**
   - Word-level: ~253k unique tokens (extremely sparse, long-tail problem)
   - Sign-level: ~16,740 unique tokens (optimal for Transformers, similar to BERT's 30k)
3. **Data Efficiency:** 4.9M signs vs 2.5M words = **2x more training tokens** from the same corpus
4. **Practical Constraint:** `value_clean` (normalized words) is completely missing for Archibab dataset

**Example:**
```
Word-level:    "a-na be-lí ù"  →  ["a-na", "be-lí", "ù"]  (3 tokens)
Sign-level:    "a-na be-lí ù"  →  ["a", "na", "be", "lí", "ù"]  (5 tokens)
```

**Why this matters:** The model can now **compose** unseen words by learning sign-to-sign patterns, rather than treating rare words as [UNK] tokens.

---

### 1.3 Why "Simplified Aeneas Twin" Architecture?

**Decision:** Implement a clean 16-layer encoder based on the Aeneas paper (Assael et al., 2025)

**Justification:** (See `yarin/justification/justification_aeneas_twin_architecture.md`)

The Aeneas model achieved state-of-the-art results on ancient text restoration (Latin epigraphy). We extract its core "torso" architecture while removing multi-modal components:

| Component | Aeneas (Original) | Our Simplified Twin |
|-----------|-------------------|---------------------|
| **Input** | Text + Image | Text Only |
| **Backbone** | 16-layer Modified T5 | 16-layer Modified T5 |
| **Positioning** | RoPE | RoPE (critical for fragmented texts) |
| **Outputs** | Restoration, Dating, Geographic | Restoration Only |

**Architecture Specs:**
- **Layers:** 16
- **d_model:** 384 (embedding dimension)
- **d_ff:** 1,536 (MLP hidden size)
- **d_kv:** 32 (per-head Q/K/V dimension)
- **Heads:** 8
- **Positional:** Rotary Embeddings (RoPE)
- **Normalization:** Pre-Norm with RMSNorm (T5-style)
- **Head:** 2-layer MLP for restoration
- **Total Parameters:** 36,705,229

**Why this is important:** We're adapting proven architecture from Latin to Akkadian, allowing direct comparison while maintaining interpretability for SAE analysis.

---

## 2. Dataset Construction

### 2.1 Sources & Scale

**Unified Akkadian Corpus Statistics:** (See `yarin/justification/data_source_summary.md`)

| Metric | Value |
|--------|-------|
| **Total Words** | 2,450,094 |
| **Total Signs** | **4,894,744** |
| **Unique Signs** | 16,740 |
| **Total Texts** | 40,429 |

**Source Breakdown:**

| Source | Tokens | % | Texts | Avg Length |
|--------|--------|---|-------|------------|
| **ORACC** | 1,385,932 | 56.6% | 14,210 | 97.5 words/text |
| **eBL** | 998,353 | 40.7% | 24,909 | 40.1 words/text |
| **Archibab** | 65,809 | 2.7% | 1,310 | 50.2 words/text |

**Comparison to Previous Work:**

| Dataset | Our Corpus (2025) | Fetaya et al. (2021) | Scale Factor |
|---------|-------------------|----------------------|--------------|
| **Total Signs** | 4.9M | 2.3M | **2.1x** |
| **Total Words** | 2.45M | 1.0M | **2.5x** |

This is a **significant scaling** of available training data for Akkadian NLP.

---

### 2.2 Data Quality & Splits

**Train/Val/Test Split** (by `fragment_id` to prevent leakage):

| Split | Words | Texts | % |
|-------|-------|-------|---|
| **Train** | 1,960,636 | 32,343 | 80% |
| **Val** | 253,798 | 4,042 | 10% |
| **Test** | 235,660 | 4,044 | 10% |

**Verification:** Zero overlapping fragment_ids between splits (confirmed programmatically)

**Data Quality:**
- 95.3% of tokens marked as "SURE" (high certainty)
- Filtered out fragmentary words containing `[`, `]`, `x`, `?` during preprocessing
- Three representations maintained:
  - `value_raw`: Original transliteration
  - `value_clean`: Normalized word
  - `value_signs`: **Sign-level tokenization** (primary training target)

---

## 3. Implementation Details

### 3.1 Data Pipeline

**Location:** `v_1/src/training/baseline/data_utils.py`

**Key Components:**

1. **Fragment Text Builder** (`build_fragment_texts`)
   - Joins `value_signs` per fragment, sorted by (`line_num`, `word_idx`)
   - Example: Reconstructs "a na be lí" from individual rows

2. **Sign Vocabulary** (`build_sign_vocabulary`)
   - 14,797 tokens = 5 special tokens + 14,792 unique signs
   - Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
   - Saved to: `v_1/data/prepared/vocab.json`

3. **MLM Dataset** (`AkkadianMLMDataset`)
   - PyTorch Dataset with dynamic masking
   - 15% masking probability
   - BERT-style 80/10/10 replacement strategy
   - Max sequence length: 512 tokens

4. **Fixed Eval Subset** (`create_eval_subset`)
   - 500 deterministic fragments (seed=42)
   - Used for pre/post training comparison
   - Ensures reproducibility

---

### 3.2 Model Implementation

**Location:** `v_1/src/training/baseline/model.py`

**Key Classes:**

1. **`AeneasConfig`** - Hyperparameter configuration
2. **`RotaryEmbedding`** - RoPE positional embeddings
3. **`RMSNorm`** - T5-style normalization (no mean centering)
4. **`AeneasAttention`** - Multi-head attention with RoPE injection
5. **`AeneasBlock`** - Transformer block (Pre-Norm architecture)
6. **`AeneasForMLM`** - Complete model with restoration head

**Critical Feature:** Hidden states extraction at layers [0, 4, 8, 12, 16] for SAE analysis

```python
outputs = model(input_ids, attention_mask,
                output_hidden_states=True,
                hidden_states_layers=[0, 4, 8, 12, 16])
```

---

### 3.3 Training Setup

**Location:** `v_1/run_training.py`

**Hardware Detection:**
- Automatically detects Apple Silicon MPS / CUDA / CPU
- Runs benchmark to find optimal batch size
- Result for my M-series Mac: batch_size=8 (11.9 samples/sec)

**Training Configuration:**
- **Optimizer:** AdamW
- **Learning Rate:** 3e-4 with Cosine Annealing
- **Batch Size:** 8 (auto-optimized for MPS)
- **Sequence Length:** 512
- **Epochs:** 10
- **Gradient Clipping:** 1.0
- **Device:** Apple Silicon MPS

**Key Features:**
- Live progress bars with tqdm
- Pre/post training artifact extraction
- Automatic checkpoint saving (init, best, last)

---

## 4. Training Results

### 4.1 Loss Progression

| Epoch | Train Loss | Val Loss | Time | Notes |
|-------|------------|----------|------|-------|
| 1 | 4.9800 | 4.5458 | 49 min | Initial steep drop |
| 2 | 4.0891 | 4.0711 | 51 min | |
| 3 | 3.6677 | 3.8166 | 51 min | |
| 4 | 3.3943 | 3.6407 | 50 min | |
| 5 | 3.1580 | 3.4887 | 50 min | |
| 6 | 2.9819 | 3.2902 | 49 min | |
| 7 | 2.8562 | 3.2377 | 49 min | |
| 8 | 2.7435 | 3.1103 | 50 min | |
| 9 | 2.6861 | 3.1124 | 50 min | |
| **10** | **2.6506** | **3.0204** | 50 min | **Best model** |

**Key Metrics:**
- **Train Loss Reduction:** 4.98 → 2.65 (47% improvement)
- **Val Loss Reduction:** 4.55 → 3.02 (34% improvement)
- **Total Training Time:** ~8.3 hours
- **Convergence:** Smooth, no overfitting (val loss still decreasing)

**Interpretation:** The model successfully learned Akkadian sign patterns through the MLM objective. The gap between train and val loss (2.65 vs 3.02) is reasonable and indicates healthy generalization.

---

### 4.2 Saved Artifacts

**Output Directory:** `v_1/models/baseline/` (~4.3 GB total)

| File | Size | Description |
|------|------|-------------|
| `baseline_init.pt` | 140 MB | Random initial weights |
| `baseline_best.pt` | 420 MB | Best checkpoint (epoch 10, val_loss=3.02) |
| `baseline_last.pt` | 420 MB | Final checkpoint |
| `baseline_pre_embeddings.pt` | 22 MB | Embedding layer before training |
| `baseline_post_embeddings.pt` | 22 MB | Embedding layer after training |
| `baseline_pre_hidden_states_layer_*.pt` | 375 MB each | Hidden states before training (5 files) |
| `baseline_post_hidden_states_layer_*.pt` | 375 MB each | Hidden states after training (5 files) |
| `training_stats.json` | <1 KB | Training metrics & config |

**Why these artifacts matter:** The pre/post embeddings and hidden states allow us to:
1. Analyze how the model's internal representations changed during training
2. Compare baseline vs MMBERT representations
3. Train Sparse Autoencoders (SAE) for interpretability in Phase 3

---

## 5. Validation & Verification

**Document:** `yarin/justification/VALIDATION_PHASE1_TRAINING.md`

For each claim, I've documented:
- **What We Claim** (the design/implementation)
- **Where to Verify in Code** (file & line numbers)
- **Proof from Logs** (actual terminal output)
- **How to Double-Check** (Python commands you can run)

**Example Validation:**

| Claim | Proof from Logs | Verification Command |
|-------|----------------|----------------------|
| Sign-level tokenization | `Vocabulary size: 14,797` | Check vocab.json |
| No data leakage | `Train: 32,343`, `Val: 4,042` | Check fragment_id overlaps |
| Hidden state extraction | 10 `.pt` files saved | `ls v_1/models/baseline/*hidden_states*.pt` |
| Training convergence | Loss: 4.98 → 2.65 | Read training_stats.json |

This validation document ensures reproducibility and transparency.

---

## 6. Technical Contributions

### 6.1 Novel Aspects

1. **Unified Multi-Source Dataset:** First work to combine eBL + ORACC + Archibab at this scale (4.9M signs)
2. **Clean Baseline Architecture:** Pure PyTorch implementation without HuggingFace dependencies, enabling full control for interpretability
3. **Sign-Level MLM Training:** Proper implementation of EvaCun 2025 tokenization methodology
4. **Pre/Post Analysis Pipeline:** Systematic extraction of embeddings and hidden states for comparative interpretability

### 6.2 Methodological Rigor

- **No data leakage:** Fragment-level splitting (not word-level)
- **Fixed eval subset:** Deterministic 500 fragments for reproducibility
- **Complete artifact preservation:** All checkpoints and intermediate representations saved
- **Documented decisions:** Every design choice justified in separate files

---

## 7. Next Steps: Phase 2 & 3

### Phase 2: MMBERT Fine-tuning (Weeks 1-2)

**Goal:** Fine-tune state-of-the-art multilingual model for comparison

**Tasks:** (See `yarin/Tasks.md` for full details)
1. Tokenization audit: Run MMBERT tokenizer on our data, check % of [UNK]
2. Build HuggingFace datasets object
3. Save pre-finetune artifacts (weights, embeddings, hidden states on same eval subset)
4. Fine-tune MMBERT with HuggingFace Trainer
5. Save post-finetune artifacts

**Research Question:** Does multilingual pretraining (MMBERT) outperform our from-scratch baseline?

---

### Phase 3: SAE Interpretability (Weeks 3-4)

**Goal:** Understand what linguistic features the models learned

**Tasks:**
1. Train Sparse Autoencoders on saved hidden states (baseline + MMBERT)
2. Analyze feature activation patterns
3. Probe for linguistic properties:
   - Suffixes vs determinatives
   - Genre markers
   - Grammatical patterns

**Research Question:** Do baseline and MMBERT learn different features? Can we identify interpretable linguistic dimensions?

---

## 8. Challenges & Solutions

### Challenge 1: Memory Constraints on MPS

**Problem:** Initial batch size 16 caused 60x slowdown due to memory swapping

**Solution:** Implemented automatic benchmark that tests batch sizes 8→128 and finds optimal throughput
- Result: batch_size=8 gives 11.9 samples/sec (best for my hardware)

### Challenge 2: Slow Training Feedback

**Problem:** No visibility into training progress during long runs

**Solution:** Added tqdm progress bars showing:
- Current batch / total batches
- Running loss
- Learning rate
- Time estimates

### Challenge 3: Ensuring Reproducibility

**Problem:** Need to prove implementation matches claims

**Solution:** Created comprehensive validation document with:
- Code references (file:line)
- Log proofs
- Verification commands

---

## 9. Files & Documentation

All work is organized in the `yarin/` folder:

| File | Purpose |
|------|---------|
| `STATUS_SUMMARY.md` | Current project status (updated after Phase 1 complete) |
| `Tasks.md` | Full implementation checklist with Phase 1/2/3 tasks |
| `PROGRESS.md` | Detailed decisions and implementation notes |
| `VALIDATION_PHASE1_TRAINING.md` | **NEW** - Proof that Phase 1 works correctly |
| `justification/justification_mlm.md` | Why MLM over causal LM |
| `justification/justification_sign_level_tokenization.md` | Why sign-level tokens |
| `justification/justification_aeneas_twin_architecture.md` | Model architecture rationale |
| `justification/data_source_summary.md` | Dataset composition & statistics |

---

## 10. Timeline

**Completed:**
- ✅ Data acquisition & processing (Dec 1-15)
- ✅ Exploratory data analysis (Dec 15-20)
- ✅ Design decisions & justifications (Dec 20-25)
- ✅ **Phase 1 implementation & training (Dec 25-29)**

**Upcoming:**
- 🔄 Phase 2: MMBERT fine-tuning (Jan 2-15)
- 📅 Phase 3: SAE interpretability (Jan 15-31)
- 📅 Thesis writing (Feb 1-28)

---

## 11. Questions for Discussion

1. **Evaluation Metrics:** Should we evaluate restoration accuracy on held-out gaps (like Fetaya et al.), or focus on perplexity/loss for Phase 1?

2. **MMBERT Comparison:** Do you think comparing to MMBERT is the right baseline, or should we also compare to GPT-style models?

3. **SAE Analysis:** Any specific linguistic features you'd like me to probe for in Phase 3 (e.g., verb conjugations, determinatives, genre markers)?

4. **Publication Strategy:** Should we aim for a workshop paper (e.g., ALP @ ACL 2026) or focus on thesis completion first?

---

## 12. References

1. **Assael et al. (2025).** *Contextualizing ancient texts with generative neural networks.* Nature. (Aeneas paper - architecture source)

2. **Fetaya et al. (2021).** *Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach.* EMNLP 2021. (Original benchmark)

3. **Gordin et al. (2025).** *EvaCun 2025 Shared Task: Lemmatization and Token Prediction for Cuneiform Languages.* ALP @ NAACL. (Tokenization methodology)

4. **Marone et al. (2025).** *MMBERT: A Modern Multilingual Encoder.* (Comparison baseline)

---

## Appendix: Key Commands

**Run training:**
```bash
# Quick test (1 epoch)
python3 v_1/run_training.py --fast

# Full training with benchmark
python3 v_1/run_training.py --benchmark --epochs 10

# Custom settings
python3 v_1/run_training.py --epochs 20 --lr 1e-4 --batch_size 32
```

**Verify artifacts:**
```bash
# Check saved files
ls -lh v_1/models/baseline/

# View training metrics
python3 -c "
import json
with open('v_1/models/baseline/training_stats.json') as f:
    print(json.dumps(json.load(f), indent=2))
"

# Check vocabulary
python3 -c "
import json
with open('v_1/data/prepared/vocab.json') as f:
    v = json.load(f)
    print(f'Vocab size: {v[\"vocab_size\"]}')
    print(f'Special tokens: {list(v[\"sign_to_id\"].items())[:5]}')
"
```

---

**End of Update**

I'm excited to discuss the results and get your feedback on the next phases!
