# Justification — No tokenizer/vocabulary expansion for the Akkadian finetune

> **Thesis claim this supports:** "Before continued-pretraining the LLMs on Akkadian we ran a
> tokenizer EDA. It showed a domain-specific vocabulary would buy almost nothing (≤8–16%
> sequence compression, saturating at ~4k pieces), our corpus is far too small to train
> thousands of new embeddings, and — decisively — the highest-value candidate tokens are
> **royal and divine names**, i.e. the exact name-leakage channel the probing pipeline controls
> for. So we kept the stock tokenizers and did plain continued pretraining."

## 1. The decision, in one sentence

We deliberately **did not expand the vocabulary / add domain BPE tokens** for the finetune
phase, because the EDA showed the gain is marginal, untrainable on ~11M tokens, and would
hard-code a known leakage channel.

## 2. The evidence (`v_1/src/finetune/eda/results/TOKENIZER_EDA.md`)

Corpus: 40,429 fragments · 2.45M words · ~11M tokens (Qwen3) / ~10M (gpt-oss).

- **Compression ceiling is tiny.** A ByteLevel BPE trained on this exact corpus saturates at
  ~4k pieces and only reaches **3.85 tokens/word** — **8% better than gpt-oss, 16% better than
  Qwen3**. Transliteration is intrinsically hyphen/space-segmented (signs are short), so there
  is little left to merge.
- **The corpus is too small to learn new embeddings.** ~10–11M tokens total cannot well-train
  5k+ fresh embedding rows.
- **The valuable candidate tokens are proper names = leakage.** The longest, highest-value
  missing pieces are dominated by **royal/divine names** (`AššurNergalilāya`, `ŠamašBēlabūa`,
  `Arbaʾilāyu`, …). "Giving those dedicated embeddings would hard-code the name-leakage channel
  the probing pipeline explicitly controls via name-masking." This is the decisive reason — it
  ties straight to [[justification_balanced_mc_protocol]] (GroupKFold-by-ruler + name-masking).

## 3. The associated finetune-setup decisions the EDA also fixed

(`TOKENIZER_EDA.md` §"Decisions this EDA supports")

- **Stock tokenizer + plain continued pretraining (CPT)**; skip the embed-align stage;
  embeddings frozen except in the full-depth (cut=0) arm.
- **Sequence length 2048 with EOS-separated packing** — covers ≥98.3% of fragments
  un-truncated, turning ~11M tokens into ~5.4k packed sequences/epoch.
- **Short-CPT budget** (~11M tokens/epoch; ~55M over 5 epochs) — low LR, watch val perplexity,
  expect fast convergence/overfit. (This is *why* perplexity moves but maximal-Spearman does
  not — see [[justification_finetune_null_result]].)
- **Leakage accounting kept explicit:** 504/1202 ORCC probe fragments fall in the unified
  *train* split (67 val, 54 test). NTP sees raw text only (no year/ruler labels), and the 37M
  MLM baseline used the same split, so the canonical split is kept *for comparability* and the
  overlap is reported (optional ablation: re-train best arm on train-minus-ORCC).

## 4. Supporting literature

- **MMBERT — "A Modern Multilingual Encoder with Annealed Language Learning"**
  (`papers/txt/Transfer Learning papers/`). Demonstrates strong multilingual coverage *without*
  per-language vocabulary surgery, supporting reliance on a shared stock tokenizer for a
  low-resource language. **[supporting.]**
- **Fetaya et al. — "Filling the Gaps in Ancient Akkadian Texts"**
  (`papers/txt/Ancient Language papers/`). Shows multilingual pretraining transfer beats
  bespoke from-scratch Akkadian modelling — i.e. exposure/objective matters more than a custom
  vocabulary in this regime. **[supporting.]** See [[justification_mlm]].

## 5. Files to pull when writing

- `v_1/src/finetune/eda/results/TOKENIZER_EDA.md` (headline table + decisions).
- `v_1/src/finetune/eda/results/tokenizer_eda.json` (full per-char/per-word stats).
- `candidate_tokens_qwen3.txt` / `candidate_tokens_gpt_oss.txt` (the rejected domain pieces —
  kept "for the record; recommendation is to NOT add them").
