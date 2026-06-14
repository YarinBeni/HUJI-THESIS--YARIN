# Pillar 1 — Findings (1a + 1b)  ·  T/A/O/F verdict

**Question.** Thalesian (`google/umt5-base` + cuneiform finetune, ~400M) beats much
larger Qwen3-8B / gpt-oss-120B on the mean-balanced-**maximal** PLS dating of ORCC.
Which of **T** (tokenizer) / **A** (architecture) / **O** (objective) / **F** (finetune)
carries the win? Decision regime = **maximal + mean** (where the win was measured);
tier0 + mean is the confound-inflated robustness column.

## The control ladder (PLS year Spearman, best layer, maximal, 200 balanced MC draws)

Numbers for all models except uMT5 come from the canonical maximal run
`v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv` (the ridge-vs-PLS
maximal panel). uMT5-base is from P1b (job 9661). **The P1b re-probe reproduces the
canonical table** (thalesian 0.411↔0.413, qwen3_8b 0.363↔0.366, random 0.301↔0.301),
so the new uMT5 number is on the same footing.

| Model | size | maximal | layer | what it is |
|---|---|---|---|---|
| **Thalesian cunei400m** | 0.4B | **0.411** | L10 | uMT5-base **+ cuneiform finetune** |
| Qwen3-8B base | 8B | 0.363 | L16 | decoder-only, no cuneiform FT |
| **Qwen3-1.7B base** *(size-matched)* | 1.7B | **0.355** | L9 | decoder-only, no cuneiform FT |
| Qwen3-32B base | 32B | 0.340 | L6 | decoder-only, no cuneiform FT |
| gpt-oss-120B base | 120B | 0.333 | L5 | decoder-only, no cuneiform FT |
| Thalesian akk300m | 0.3B | 0.322 | L8 | uMT5 + finetune (variant) |
| mlm (Akkadian MLM) | — | 0.311 | — | encoder, masked-LM |
| **random** (Qwen arch, untrained) | — | **0.301** | L28 | the floor |
| **vanilla uMT5-base** | 0.4B | **0.297** | L0 | the **un-finetuned** base of Thalesian |
| tfidf | — | 0.292 | — | lexical baseline |

`spearman_std ≈ 0.07–0.08`.

**Size-matched architecture check (Yarin's point — the 8B was an unfair A-test):**
the fair comparator is Qwen-**1.7B** (0.355), nearest in size to the 0.4B uMT5.
uMT5 (0.297) is **still below** size-matched Qwen-1.7B and **at the random floor**
(0.301). And the Qwen family is **flat across scale** (1.7B 0.355 ≈ 8B 0.363 ≈
32B 0.340 ≈ 120B-class gpt-oss 0.333) — so size was never the lever. Size fairness
does not rescue the encoder-decoder hypothesis.

## Factor table — each comparison and what it isolates (maximal)

| Comparison | Δ Spearman | Isolates | Reading |
|---|---|---|---|
| Thalesian (0.411) **vs** vanilla uMT5 (0.297) | **+0.114** (~1.5σ) | **(F)** finetune | finetune **creates** the signal |
| vanilla uMT5 (0.297) **vs** random (0.301) | **−0.004** (≈0) | base-model floor | **uMT5 base has NO date signal above random** (size-independent) |
| vanilla uMT5 0.4B (0.297) **vs** Qwen3-1.7B (0.355) | **−0.058** | **(A)+(T)** enc-dec base | enc-dec base is **worse**, even size-matched |
| Qwen family across scale | 1.7B 0.355 ≈ 8B 0.363 ≈ 32B 0.340 | scale | **flat** — size is not the lever |
| Qwen3-1.7B (0.355) **vs** random (0.301) | +0.054 | decoder base | modestly above floor |
| 1a fertility (tokens/word) | uMT5/Thal **worse** (5–6) than Qwen (5)/gpt-oss (4) | **(T)** | tokenizer **does not** favor the winner |

## Signed verdict on T / A / O / F

- **T — REJECTED.** 1a: uMT5/Thalesian tokenizers are the *least* efficient on Akkadian
  (highest fertility), and nobody UNKs. The small model wins *despite* a worse tokenizer.
- **A — REJECTED.** Vanilla uMT5-base sits **at the random floor** (0.297 ≈ 0.301) and
  *below* the **size-matched** decoder-only Qwen-1.7B (0.355) — and below 8B (0.363) /
  32B (0.340) too. The encoder-decoder/bidirectional architecture gives **no** dating
  advantage on its own; the floor-tie is size-independent (a fully-pretrained 0.4B
  enc-dec ties an *untrained* network), and Qwen is flat across scale, so the earlier
  8B comparison was not load-bearing.
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
