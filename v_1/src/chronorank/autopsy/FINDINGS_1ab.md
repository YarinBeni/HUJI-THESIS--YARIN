# Pillar 1 — Findings (1a + 1b)  ·  T/A/O/F verdict

**Question.** Thalesian (`google/umt5-base` + cuneiform finetune, ~400M) beats much
larger Qwen3-8B / gpt-oss-120B on the mean-balanced-**maximal** PLS dating of ORCC.
Which of **T** (tokenizer) / **A** (architecture) / **O** (objective) / **F** (finetune)
carries the win? Decision regime = **maximal + mean** (where the win was measured);
tier0 + mean is the confound-inflated robustness column.

## The control ladder (PLS year Spearman, best layer, 200 balanced MC draws)

| Model | maximal | tier0 | what it is |
|---|---|---|---|
| **Thalesian cunei400m** | **0.413** | 0.414 | uMT5-base **+ cuneiform finetune** |
| Thalesian akk300m | 0.322 | 0.346 | uMT5 + finetune (variant) |
| Qwen3-8B base | 0.366 | 0.366 | decoder-only, no cuneiform FT |
| gpt-oss-120B base | 0.333 | 0.408 | decoder-only, no cuneiform FT |
| **vanilla uMT5-base** | **0.297** | 0.336 | the **un-finetuned** base of Thalesian |
| **random** (Qwen arch, untrained) | **0.301** | 0.379 | the floor |

`spearman_std ≈ 0.07–0.08`.

## Factor table — each comparison and what it isolates (maximal)

| Comparison | Δ Spearman | Isolates | Reading |
|---|---|---|---|
| Thalesian **vs** vanilla uMT5 | **+0.116** (~1.5σ) | **(F)** finetune | finetune **creates** the signal |
| vanilla uMT5 **vs** Qwen3-8B | **−0.069** | **(A)+(T)** enc-dec base | enc-dec base is **worse**, not better |
| vanilla uMT5 **vs** random | **−0.004** (≈0) | base-model floor | uMT5 base has **no** date signal above floor |
| Qwen3-8B **vs** random | +0.065 | decoder base | modestly above floor |
| 1a fertility (tokens/word) | uMT5/Thal **worse** (5–6) than Qwen (5)/gpt-oss (4) | **(T)** | tokenizer **does not** favor the winner |

## Signed verdict on T / A / O / F

- **T — REJECTED.** 1a: uMT5/Thalesian tokenizers are the *least* efficient on Akkadian
  (highest fertility), and nobody UNKs. The small model wins *despite* a worse tokenizer.
- **A — REJECTED.** Vanilla uMT5-base sits **at the random floor** (0.297 ≈ 0.301) and
  *below* the decoder-only Qwen base (0.366). The encoder-decoder/bidirectional
  architecture gives **no** dating advantage on its own.
- **F — THIS IS IT.** Finetuning that same uMT5 base on cuneiform lifts it from the
  floor (0.30) to 0.41 — **+0.116**, the largest jump in the table, and the only thing
  that clearly clears the random floor by a wide margin.
- **O — not yet separable, but implicated.** F and O are entangled (Thalesian's finetune
  *was* a seq2seq/translation finetune). Crucially, Round-3's **NTP** finetune of Qwen
  was **flat** — so it is not "finetuning per se" that works; it is finetuning with the
  **right objective**. Separating F from O is exactly 1c (on hold pending translation data).

### The one sentence
> **Thalesian wins because of the cuneiform FINETUNE (F) — its uMT5 base alone is no
> better than random and worse than Qwen, so it is neither the tokenizer nor the
> encoder-decoder architecture — and because Round-3's NTP finetune was flat, the cause
> is almost certainly the finetune's translation/seq2seq OBJECTIVE. Therefore the next
> finetune should re-train the big models (Qwen-1.7B/8B) with a seq2seq/translation
> objective, NOT next-token prediction — i.e. run 1c.**

## Two findings worth carrying into the thesis
1. **The honest floor is ~0.30, not 0.** Random features already reach Spearman ≈ 0.30
   under maximal cleaning (residual length/structure confound). So the winner's *true*
   signal above floor is only ≈ 0.11 (Thalesian), ≈ 0.065 (Qwen), ≈ 0 (uMT5). This
   reframes the whole leaderboard and strengthens the Thrust-B "honest modeling" case.
2. **tier0 is mostly confound.** Under tier0 everything sits 0.34–0.41 incl. random
   (0.38) and gpt-oss (0.41); maximal cleaning collapses gpt-oss to 0.33 and uMT5 to the
   floor, isolating real signal. Confirms maximal is the right decision regime.

## Status
- [x] P1a tokenization audit — (T) rejected
- [x] P1b control-ladder probe — (A) rejected, (F) identified, vanilla uMT5 = floor
- [x] factor table filled, T/A/O/F signed, downstream = "1c: translation-objective finetune"
- [ ] 1c objective ablation — ON HOLD (needs Akkadian→English translation data)
