# Tokenizer pre-EDA — Akkadian NTP fine-tune (Task 5)

**Date:** 2026-06-10 · **Script:** `v_1/src/finetune/eda/tokenizer_eda.py` (runs locally, no GPU)
**Corpus:** unified train+val+test, fragment texts = space-joined `value_clean` (fallback `value_raw`), tier0-cleaned.
40,429 fragments · 2,450,094 words · 4,894,736 signs · 18.77M chars.

## Headline numbers

| | Qwen3 (151k vocab, shared 1.7B/8B/32B) | gpt-oss (o200k, shared 20b/120b) |
|---|---|---|
| Total corpus tokens | **11,252,825** | **10,286,022** |
| Tokens / word | 4.59 | 4.20 |
| Tokens / sign | 2.30 | 2.10 |
| Chars / token | 1.67 | 1.83 |
| Unique token ids used (whole corpus) | **2,676** | **3,636** |
| Byte-fallback tokens (decode to partial UTF-8) | 8.3 % | 9.2 % |
| Top-2000 word forms that are 1 token | 182 (9 %) | 215 (11 %) |
| Top-2000 word forms needing ≥3 tokens | 1,543 | 1,407 |
| Fragments > 2048 tokens | 680 (1.7 %) | 605 (1.5 %) |
| Median / p99 fragment length (tokens) | 82 / 2,957 | 76 / 2,680 |

Examples (freq = corpus count):
`ina` (80.6k): Qwen ` in`+`a`, gpt-oss **1 token**. `a-na` (36.4k): both ` a`+`-na`.
`LUGAL` (26.3k): both 3 tokens. `ša₂` (51.9k): both 3 tokens. `u₃` (14.7k): `u` + **2 raw bytes** (₃ has no token).
Diacritics š ṣ ṭ ḫ and ₂ each have a dedicated token in both vocabs; ₃ ₄ fall to byte pairs (and ḫ/ʾ in gpt-oss).

## What a domain BPE could buy (vocab-expansion ceiling)

ByteLevel BPE trained on this exact corpus:

| domain vocab | tokens/word |
|---|---|
| 4,000 | 3.859 |
| 8,000 | 3.851 |
| 16,000 | 3.851 |

→ Saturates at ~4k pieces; best case **3.85 tokens/word**, i.e. only **8 % better than gpt-oss, 16 % better than Qwen3**.
Transliteration is intrinsically hyphen/space-segmented (signs are short), so there is little left to merge.
Of the ≥3-char domain pieces, 904 (Qwen) / 1,504 (gpt-oss) already exist as single tokens; ~5.3–5.9k are missing
(`candidate_tokens_*.txt`) — but the longest, highest-value candidates are dominated by **royal/divine names**
(`AššurNergalilāya`, `ŠamašBēlabūa`, `Arbaʾilāyu`, …). Giving those dedicated embeddings would hard-code the
name-leakage channel the probing pipeline explicitly controls via name-masking.

## Decisions this EDA supports

1. **No vocabulary expansion.** Gain ≤ 8–16 % sequence compression, corpus is only ~10–11M tokens (too little to
   train 5k+ new embeddings well), and the candidate tokens are leakage-prone proper names. Plain continued
   pretraining (CPT) with the stock tokenizer; skip the embed-align stage entirely. Embeddings stay frozen except
   in the full-depth (cut=0) ablation arm.
2. **Sequence length 2048 with packing.** Covers ≥ 98.3 % of fragments un-truncated; pack fragments with EOS
   separators so the ~11M-token corpus turns into ~5.4k packed sequences per epoch.
3. **Training budget is small by construction.** ~11M tokens/epoch → even 5 epochs ≈ 55M tokens. This is a
   short-CPT regime: low LR, watch val perplexity per epoch, expect fast convergence/overfit.
4. **gpt-oss is the (slightly) better-suited tokenizer** out of the box (4.20 t/w, `ina` is a single token), but
   both are byte-fragmenting Akkadian heavily — exactly the "Qwen reads bytes" point from the 03.06 notes.
5. **Leakage accounting:** 504/1202 ORCC probing fragments are inside the unified *train* split (67 in val, 54 in
   test). NTP sees raw text only — no year/ruler labels — and the 37M MLM baseline was trained on the same split,
   so we keep the canonical split for comparability and report the overlap. Optional ablation: re-train best arm
   on train-minus-ORCC.

## Files

- `tokenizer_eda.json` — all stats (incl. per-char tokenization, top-word examples, quantiles)
- `candidate_tokens_qwen3.txt` / `candidate_tokens_gpt_oss.txt` — domain-BPE pieces missing from each vocab
  (kept for the record; recommendation is to NOT add them)
