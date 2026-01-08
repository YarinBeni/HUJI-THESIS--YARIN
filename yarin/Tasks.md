### Akkadian Restoration (MLM) — Implementation Task List

**Goal**: Train (1) a clean PyTorch “torso” baseline (MLM) on the unified sign-level dataset in `v_1/`, then (2) fine-tune **MMBERT** with Hugging Face for comparison, saving **weights + embeddings/hidden-states before/after**, and finally (3) run **SAE interpretability**.

**Key constraints**
- **Objective**: **MLM** (masking) — *not* causal LM.
- **Data**: Use `v_1/data/processed/unified/{train,val,test}.parquet`.
- **Analysis artifacts**: Save **pre/post** weights + embeddings + selected layer hidden states on a fixed eval subset for comparability.

---

### Phase 0 — Decisions to lock ✅ FINALIZED

- [x] **0.1 MLM masking granularity**
  - **Decision**: Mask **individual signs** (token-level).
  - Each sign from `value_signs` (split on spaces) is a separate token.
  - Example: `"a na be li"` → tokens `["a", "na", "be", "li"]` → mask individual signs.
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

**Deliverable**: ✅ Decisions documented above and in justification files.

---

### Phase 1 — Baseline PyTorch "torso" MLM (clean model)

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

#### 1A. Dataset view for training ✅ COMPLETE
- [x] **1A.1 Create a "text per fragment" builder**
  - **Implementation**: `v_1/src/data_utils.py::build_fragment_texts()`
  - Joins `value_signs` per fragment, sorted by (`line_num`, `word_idx`)
- [x] **1A.2 Build token vocabulary for signs**
  - **Implementation**: `v_1/src/data_utils.py::build_sign_vocabulary()`
  - Vocabulary: 14,797 tokens (5 special + 14,792 signs)
  - Saved to: `v_1/data/prepared/vocab.json`
- [x] **1A.3 Create PyTorch Dataset + Collator for MLM**
  - **Implementation**: `v_1/src/data_utils.py::AkkadianMLMDataset`
  - 15% masking, BERT-style 80/10/10 replacement
- [x] **1A.4 Fixed eval subset**
  - 500 deterministic fragments (seed=42)
  - Saved to: `v_1/data/prepared/eval_subset.parquet`

**Deliverables** ✅
- `v_1/src/data_utils.py` - Core data utilities
- `v_1/src/01_prepare_data.py` - Data preparation script
- `v_1/data/prepared/` - Prepared data directory

#### 1B. Model torso implementation (PyTorch) ✅ COMPLETE
- [x] **1B.1 Implement Simplified Aeneas Twin**
  - **Implementation**: `v_1/src/model.py::AeneasForMLM`
  - 16 layers, d_model=384, d_kv=32, RoPE, Pre-Norm
  - ~37M parameters
- [x] **1B.2 Add "return hidden states" support**
  - **Implementation**: `output_hidden_states=True, hidden_states_layers=[0,4,8,12,16]`
  - Returns Dict[layer_idx → tensor]

**Deliverables** ✅
- `v_1/src/model.py` - Model implementation

#### 1C. Training + checkpointing (baseline) ✅ COMPLETE
- [x] **1C.1 Training script**
  - **Implementation**: `v_1/src/02_train.py`
  - Saves `baseline_init.pt`, `baseline_best.pt`, `baseline_last.pt`
  - AdamW optimizer with cosine annealing
- [x] **1C.2 Baseline embedding/hidden-state dumps (pre/post)**
  - Extracts embeddings and hidden states for layers [0, 4, 8, 12, 16]
  - Saves pre/post training artifacts

**Deliverables** ✅
- `v_1/src/02_train.py` - Training script
- Output directory: `v_1/models/baseline/`

**To run training:**
```bash
# Quick test (1 epoch)
python3 v_1/run_training.py --fast

# Full training (10 epochs)
python3 v_1/run_training.py --epochs 10

# Custom settings
python3 v_1/run_training.py --epochs 20 --lr 1e-4 --batch_size 32
```

---

### Phase 2 — MMBERT fine-tuning (Hugging Face, MLM)

#### 2A. Sanity-check tokenizer coverage
- [ ] **2A.1 Tokenization audit**
  - Run MMBERT tokenizer over a sample of `text_signs`
  - Report: % `[UNK]` (if applicable), average pieces per sign/word, max length stats
  - Decide if preprocessing tweaks are needed (spacing, boundary token, normalization)

**Deliverable**: small report (markdown or txt) with tokenization stats.

#### 2B. Fine-tuning pipeline
- [ ] **2B.1 Build HF `datasets` object for train/val/test**
  - One row per fragment (or chunked sequences)
- [ ] **2B.2 Implement MLM data collator**
  - Use HF collator or custom masking to match your baseline
- [ ] **2B.3 Save “pre-finetune” artifacts**
  - Save MMBERT model weights before updates
  - Dump embeddings + hidden states on the same fixed eval subset (same layers schedule)
- [ ] **2B.4 Fine-tune**
  - Train with HF Trainer/Accelerate
  - Save checkpoints and final model
- [ ] **2B.5 Save “post-finetune” artifacts**
  - Save weights after fine-tune
  - Dump embeddings + hidden states again (same eval subset, same layer indices)

**Deliverables**
- `mmbert_pre_ft_state_dict.pt`, `mmbert_post_ft_state_dict.pt`
- `mmbert_pre_hidden_states_layers_*.pt`, `mmbert_post_hidden_states_layers_*.pt`
- HF output dir with config/tokenizer (if applicable)

---

### Phase 3 — SAE interpretability (baseline + MMBERT)

#### 3A. Standardize activation dataset
- [ ] **3A.1 Choose activation source**
  - Use hidden states you saved in Phase 1C/2B, or re-extract on demand
- [ ] **3A.2 Create a consistent on-disk schema**
  - Include: model id, layer id, token ids, attention mask, mapping back to `fragment_id`

#### 3B. SAE training
- [ ] **3B.1 Train SAE per chosen layer (baseline torso)**
  - Start with a middle-ish layer + a late layer
  - Track sparsity, dead features, reconstruction loss
- [ ] **3B.2 Train SAE per chosen layer (MMBERT)**
  - Same procedure for comparability

#### 3C. Feature analysis
- [ ] **3C.1 Top-activating examples per feature**
- [ ] **3C.2 Feature → linguistic probe notebooks/scripts**
  - e.g., suffixes, determinatives, genre markers, editorial/certainty effects (if retained)

**Deliverables**
- SAE weights + metrics json per (model, layer)
- Lightweight feature inspection utilities (histograms, top-k contexts)

---

### References inside this repo (useful templates)
- **Baseline torso + SAE pipeline (older)**: `v_0/RESTORATION_PROJECT_PLAN.md`
- **Torso model implementation example**: `v_0/src/modeling_restoration.py`
- **Torso training example**: `v_0/src/03_train_torso.py`
- **SAE utilities (older)**: `v_0/src/06_sae_memory_optimized.py`, `v_0/src/07_inspect_sae.py`


