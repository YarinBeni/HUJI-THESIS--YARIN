# Akkadian Thesis — Research Log

Condensed record of milestones, key decisions, and results. Useful for writing the paper/thesis.

---

## Phase Timeline

| Phase | Status | Date |
|-------|--------|------|
| Phase 0: Design Decisions | COMPLETE | Dec 2025 |
| Phase 1: Baseline MLM Training | COMPLETE | Dec 29, 2025 |
| Phase 1.5: Evaluation Corpus | COMPLETE | Jan 25, 2026 |
| Phase 2 / Track A: LLM Baseline Pipeline | IN PROGRESS | Jan–Mar 2026 |
| Track B: Embedding Manifold Analysis | PENDING | Apr 2026 |
| Track C: SAE Interpretability | PENDING | Apr–May 2026 |

**Note (Mar 2026):** Advisor decided to shift from training from random weights to fine-tuning pre-trained LLMs (Gemma-2-9B-IT, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct). The baseline MLM artifacts (Phase 1) are no longer part of the experimental plan.

---

## Phase 0: Design Decisions (Dec 2025)

### Why MLM (BERT-style) over Causal LM?
- Restoration task is bidirectional — scholars use context from both sides of a gap
- MMBERT (multilingual encoder) outperforms decoder-only on understanding tasks in low-resource settings
- Fetaya et al. (2021): zero-shot multilingual BERT outperformed monolingual Akkadian models
- Details: `justification/justification_mlm.md`

### Why Sign-Level Tokenization?
- Word-level: ~253k unique tokens (too sparse)
- Sign-level: ~16,740 unique tokens (optimal, similar to BERT's 30k)
- Sign-level gives 2x more training tokens (4.9M signs vs 2.5M words)
- SOTA precedent: EvaCun 2025 uses transliterated signs as minimal units
- `value_clean` completely missing for Archibab — sign-level is the only option
- Details: `justification/justification_sign_level_tokenization.md`

### Why "Simplified Aeneas Twin" Architecture?
- Adapted from Assael et al. (2025) — SOTA on ancient text restoration (Latin epigraphy)
- Removed multi-modal components (image input, dating/geographic heads), kept the core encoder
- 16-layer Modified T5 with RoPE positional embeddings
- Details: `justification/justification_aeneas_twin_architecture.md`

---

## Phase 1: Baseline MLM Training (Dec 2025)

### Dataset

| Source | Tokens | % |
|--------|--------|---|
| ORACC | 1,385,932 | 56.6% |
| eBL | 998,353 | 40.7% |
| Archibab | 65,809 | 2.7% |
| **Total** | **2,450,094 words / 4,894,744 signs** | |

- Train/Val/Test: 80/10/10 split at fragment level (no leakage)
- Unified dataset: `v_1/data/processed/unified/`

### Model Architecture
- **Name**: Simplified Aeneas Twin
- **Params**: 36,705,229 (37M)
- **Layers**: 16, d_model: 384, d_ff: 1536, heads: 8
- **Positional encoding**: RoPE
- **Normalization**: Pre-norm with RMSNorm (T5-style)
- **Task**: Masked Language Modeling (15% masking, BERT 80/10/10 strategy)

### Training Results
- **Device**: Apple Silicon MPS
- **Epochs**: 10 (~50 min/epoch = ~8.3 hours total)
- **Batch size**: 8

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 4.9800 | 4.5458 |
| 2 | 4.0891 | 4.0711 |
| 3 | 3.6677 | 3.8166 |
| 4 | 3.3943 | 3.6407 |
| 5 | 3.1580 | 3.4887 |
| 6 | 2.9819 | 3.2902 |
| 7 | 2.8562 | 3.2377 |
| 8 | 2.7435 | 3.1103 |
| 9 | 2.6861 | 3.1124 |
| **10** | **2.6506** | **3.0204** ← best model |

**Note:** These artifacts (~4.3GB) are no longer in git. Model is not part of the current experimental plan.

---

## Phase 1.5: Evaluation Corpus (Jan 2026)

### Data from Chunrong Ni (Jan 20–23, 2026)
Received 3 normalized CSVs for temporal classification:

| File | Period | Texts | Words |
|------|--------|-------|-------|
| `archibab_nor.csv` | Old Babylonian (~1800 BCE) | 1,526 | 80,059 |
| `oracc_let_adm_nor.csv` | Neo-Assyrian (9–7 cent BCE) | 4,007 | 229,191 |
| `lbl_nor.csv` | Late Babylonian (~600 BCE) | 1,044 | 74,169 |

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Include Neo-Assyrian? | YES (all 3 groups) | Originally planned to skip — included for richer comparison |
| ORACC domain filter | Letters only (NALet) | Exclude administrative docs (NAAdm) for cleaner task |
| Archibab domain merges | 4 merges applied | Fix typos, standardize labels (e.g., `inconnu, lettre cassée` → Unknown) |
| Unknown domain texts | Removed (~29 texts, <1%) | Clean labels needed for evaluation |

### Final Corpus
- **Total texts after filtering**: 4,957
- **File**: `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet`
- Details: `justification/evaluation_corpus_size_5005_texts.md`, `justification/chunrong_data_cleaning_decisions.md`

---

## Track A: LLM Baseline (Jan–Mar 2026)

### Goal
Zero-shot classification of Akkadian letters by temporal period (OB / NA / LB) using general-purpose LLMs via OpenRouter API.

### Pipeline (built Jan 28, 2026)
- `v_1/src/preprocessing/06_create_test_letters_copra.py` — merge + filter corpora
- `v_1/src/evaluation/01_prepare_texts.py` — reconstruct texts from word-level rows
- `v_1/src/evaluation/02_llm_baseline.py` — OpenRouter API calls, resume-safe caching
- `v_1/src/evaluation/03_aggregate_results.py` — merge all model predictions

### Initial Test Results (10 texts, GPT-OSS-20B, Jan 2026)
- Parse rate: 90% (9/10)
- Period accuracy: **22.2%** (2/9 correct)
- Model biased toward Neo-Assyrian (6/9 misclassified as NA)
- Token usage: ~24k per 10 texts (heavy: reasoning models think before answering)
- Fix needed: increase `MAX_COMPLETION_TOKENS` from 300 → 4096

### Key Technical Fix
Reasoning models (GPT-OSS-20B) put answer in `message['reasoning_details']` not `content`. Parse rate improved from ~60% → 90% after fix.

### Model Plan for Track A
Target models (pre-trained SAEs available for ★ models used in Tracks B/C):

| Category | Models |
|----------|--------|
| Free tier | gpt-oss-20b, gpt-oss-120b, llama-4-maverick, gemini-2.5-pro-free, deepseek-r1-free |
| Open-source (paid) | ★ Qwen2.5-7B-Instruct, ★ Llama-3.1-8B-Instruct, Qwen-32B/72B |
| Commercial | GPT-4o, Claude Sonnet, Gemini 2.0 Flash |
| Fine-tune targets | ★ Gemma-2-9B-IT |

### Compute
Running on Schmidt Sciences cluster (64x H100 80GB). API calls are CPU-only (no GPU needed for Track A).

---

## Key References
- Fetaya et al. (2021) — "Filling the Gaps in Ancient Akkadian Texts" (EMNLP 2021)
- Assael et al. (2025) — Aeneas model, SOTA ancient text restoration
- Gurnee & Tegmark (ICLR 2024) — Linear representation of space and time in LLMs
- Gemma Scope (2024) — Pre-trained SAEs for Gemma-2 models
