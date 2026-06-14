# Justification — The Thalesian autopsy (why the small model wins: finetune, not tokenizer/architecture)

> **Thesis claim this supports:** "Thalesian (a 0.4B `google/umt5-base` finetune) beats much
> larger Qwen3/gpt-oss on confound-controlled Akkadian dating **because of the cuneiform
> finetune itself — specifically its translation/seq2seq objective — not its tokenizer and not
> its encoder-decoder architecture.** The vanilla uMT5 base has no dating signal above the
> random floor; finetuning is what creates it. Therefore the productive next step is to finetune
> with a translation/seq2seq objective, not next-token prediction."

## 1. The decision/finding, in one sentence

We ran a **control ladder** — probe the *un*-finetuned `google/umt5-base` under the identical
`pls__mc_balanced_maximal` protocol alongside Thalesian and the decoder-only models — so each of
the four candidate causes (Tokenizer / Architecture / Objective / Finetune) is isolated by one
controlled comparison; the ladder rejects **T** and **A**, and pins the win on **F** (the
finetune), implicating **O** (the objective) once read together with the NTP-null result.

## 2. The evidence

### 2a. The control ladder (maximal, balanced, best-layer year-PLS Spearman)

Canonical per-layer numbers: `v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv`;
vanilla uMT5 from Pillar-1 job 9661 (`v_1/src/chronorank/autopsy/results/ladder_table.csv`). The
re-probe reproduces the canonical table (Thalesian 0.411↔0.413, qwen3_8b 0.363↔0.366, random
0.301↔0.301), so the new uMT5 number is on the same footing.

| Model | size | maximal Spearman | what it isolates |
|---|---|---|---|
| **Thalesian cunei400m** (uMT5 + cuneiform FT) | 0.4B | **0.411** | — (the winner) |
| Qwen3-8B base | 8B | 0.363 | decoder-only |
| **Qwen3-1.7B base** (size-matched) | 1.7B | **0.355** | decoder-only, fair A-test |
| Qwen3-32B base | 32B | 0.340 | scale |
| gpt-oss-120B base | 120B | 0.333 | scale |
| Thalesian akk300m (variant) | 0.3B | 0.322 | — |
| **random** (Qwen arch, untrained) | — | **0.301** | the floor |
| **vanilla uMT5-base** (NO finetune) | 0.4B | **0.297** | the base of Thalesian |

`spearman_std ≈ 0.07–0.08`. The three controlled reads:

| Comparison | Δ | Isolates | Verdict |
|---|---|---|---|
| Thalesian (0.411) − vanilla uMT5 (0.297) | **+0.114** | **(F)** finetune | finetune **creates** the signal |
| vanilla uMT5 (0.297) − random (0.301) | ≈0 | base floor | uMT5 base has **no** signal above random (size-independent) |
| vanilla uMT5 0.4B (0.297) − Qwen3-1.7B (0.355) | −0.058 | **(A)+(T)** | enc-dec base is **worse**, even size-matched |

**(A) rejected — the fair, size-matched test.** uMT5 (0.4B) is below the nearest-size decoder
Qwen3-1.7B (0.355) and tied to the random floor; the Qwen family is flat across scale
(1.7B 0.355 ≈ 8B 0.363 ≈ 32B 0.340 ≈ gpt-oss 0.333), so the encoder-decoder/bidirectional
architecture confers no dating advantage and size was never the lever.

**Depth profile makes it visual** (`factor_ladder_layerwise.png`): vanilla uMT5 peaks at the
**embedding layer (L0, 0.297)** and *decays to ~0.18 below the floor* in deeper encoder layers —
i.e. its only date signal is lexical and its pretraining-learned layers wash it out — whereas the
finetune builds a representation that rises with depth to 0.41 (best layer L10).

### 2b. Tokenization audit (factor T), `v_1/src/chronorank/autopsy/tokenization_audit.csv`

Fertility = tokens per Akkadian word (lower = more efficient). Across 5 corpora
(orcc/seal/letters/archibab/oracc_1mill):

| tokenizer | ORCC fertility | char-probe (tok/diacritic) | UNK |
|---|---|---|---|
| gpt-oss-120B | **4.43** | 1.17 | 0 |
| Qwen3-8B | 5.06 | **1.00** | 0 |
| uMT5-base | 5.22 | 1.72 | 0 |
| **Thalesian** | **6.22** | 1.72 | 0 |

**(T) rejected.** uMT5/Thalesian have the **least** efficient Akkadian tokenizers (highest
fertility, most diacritic fragmentation) and nobody UNKs — the small model wins *despite* a worse
tokenizer, so the tokenizer cannot be the cause. (Note: Thalesian did not expand the vocab —
[[justification_no_vocab_expansion]] — yet its segmentation is even coarser than vanilla uMT5.)

## 3. Why the experiment was designed this way (so the attribution is credible)

- **Public base = clean controls.** Thalesian = `google/umt5-base` + finetune, and the base is
  public, so probing the un-finetuned base isolates **F** (Thalesian − uMT5) from the
  architecture/tokenizer bundle (uMT5 − Qwen) with no training required.
- **Identical protocol.** Same seed-42 splits, GroupKFold by ruler, 200 MC balanced draws,
  maximal cleaning, mean pooling — see [[justification_balanced_mc_protocol]],
  [[justification_maximal_cleaning_regime]] — so Δ is attributable to the one varied factor.
- **Apples-to-apples extraction.** Vanilla uMT5 was extracted with the *same* seq2seq-encoder
  extractor that produced the on-disk Thalesian activations (`round2_phase3/extract_enc_activations.py`,
  `UMT5ForConditionalGeneration`, 13 hidden states, mean-pooled encoder) — not the decoder path.
- **Size-matched A-test.** Including Qwen3-1.7B (nearest size to 0.4B uMT5) removes the
  size-vs-architecture confound in the uMT5-vs-Qwen comparison.
- **Random floor reported.** Random features already reach Spearman ≈ 0.30 under maximal
  (residual length/structure), so "above floor" — not "above zero" — is the bar; uMT5 is at it.

## 4. Why this is theoretically expected (the thesis argument)

This dovetails with the NTP-null result ([[justification_finetune_null_result]]): NTP finetuning
of the big models is flat (Δ≈0 at maximal), yet Thalesian's finetune lifts uMT5 by +0.114. The
difference between them is the **objective** — Thalesian's finetune was translation /
transliteration / script-conversion (seq2seq), which forces the model to align surface
orthography with meaning, whereas NTP only rewards local next-sign prediction. So "finetuning
per se" is not sufficient (NTP fails); it is finetuning with a **chronology-bearing, seq2seq
objective**. F and O are not fully separable from on-disk data alone — that separation is **1c**
(finetune one model NTP-vs-translation on the same data), on hold pending Akkadian→English
parallel data. The actionable conclusion is unchanged: the next finetune should use a
**translation/seq2seq objective**, or a purpose-built bidirectional/translation backbone.

## 5. Supporting literature

- **Thalesian `cuneiformBase-400m` model card** — confirms the winner is `umt5-base` + cuneiform
  translation/transliteration tasks (encoder-decoder, seq2seq). **[direct — defines the factors.]**
- **mT5 / uMT5 (Xue 2021; Chung 2023)** — span-corruption, bidirectional encoder, multilingual;
  grounds factors A and O and what the vanilla base does/doesn't encode. **[direct.]**
- **Gutherz et al. 2023 (Akkadian→English NMT, BLEU≈37)** — establishes that a real
  Akkadian→English signal exists, making the translation-objective hypothesis (O) realistic and
  1c implementable. **[supporting.]**
- **Gurnee & Tegmark, "LMs Represent Space and Time"** — temporal structure is a property of
  *representations*; we show it is the finetune (objective), not scale/architecture, that
  instills it for Akkadian. **[supporting.]** See also [[justification_mlm]] (objective >
  exposure in low-resource ancient languages).

## 6. Figures & tables to pull when writing

- **Factor ladder (headline):** `v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png`
- **Depth profile:** `v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png`
- **Tokenizer (factor T):** `figures/fertility_by_corpus.png`, `figures/fertility_hist.png`
- **Tables:** `results/ladder_table.csv`, `results/tokenization_audit.csv`,
  `v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv`
- **Full writeup:** `v_1/src/chronorank/autopsy/FINDINGS_1ab.md`
- **Probe summaries:** `results/probes/umt5_base_pls__mc_balanced{,_maximal}__summary.json`
