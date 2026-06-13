# Pillar 1 — The Thalesian autopsy (why does it win?)  ★ THE FOCUS — DO FIRST

> **Agent brief.** This is the priority of Round 4. Thalesian is the only real winner of the
> mean-balanced-maximal PLS experiment, so naming *what* makes it win is the highest-value work
> we can do — it is a finding on its own (what do LLMs encode about historical time?) **and** it
> is actionable: the cause tells us how to build a better backbone (see "Why this matters
> downstream"). Read `README.md` first.
>
> **CURRENT SCOPE (Yarin, 2026-06-14): do 1a + 1b only.** Both need no training and run on
> activations already on disk — they give the decisive narrowing of the four causes. **1c and 1d
> are ON HOLD** because they require Akkadian→English translation data Yarin hasn't provided yet;
> do not build them now. Yarin will say when translations are available, and then we un-hold 1c/1d.

## Goal — disentangle the four candidate causes

Thalesian = **Google `uMT5-base`** (public) **+ cuneiform finetuning**. That public-base fact is
the gift that makes this clean: we can probe the *un-finetuned* base and isolate each factor.
Four candidate explanations for the win:

- **(T) Tokenization** — cuneiform/transliteration-aware vocab → fewer, more meaningful units.
- **(A) Architecture** — encoder-decoder with a **bidirectional encoder** (what we mean-pool) vs
  decoder-only causal models (Qwen, gpt-oss).
- **(O) Objective** — seq2seq / span-corruption / translation supervision vs next-token prediction.
- **(F) The cuneiform finetune itself** — domain adaptation, regardless of objective.

The deliverable is a **signed conclusion on which of T/A/O/F carries the win**, backed by a
factorial probing table. This is not a vague "it's domain-specific" — it's "factor X, here's the
controlled comparison that isolates it."

## Why this matters downstream (the actionable payoff — keep this in view)

- If **(O) objective** wins → redo finetuning of the *big* models with a seq2seq/translation
  objective (not NTP) to get a **better frozen backbone**; or train a purpose-built
  bidirectional/translation model. This is the most exciting path: it would explain why the
  Round-3 NTP finetune was flat.
- If **(A) architecture** wins → take a *bigger* encoder-decoder (mT5-XL / uMT5-XL) and test whether
  scale *within the right architecture* finally helps — the scaling result may reverse.
- If **(T) tokenization** wins → revisit the vocab-expansion decision (`finetune/eda/TOKENIZER_EDA.md`
  said it was skipped) for the big models.
- If **(F) finetune-per-se** wins → the question becomes why Qwen's NTP finetune failed but
  Thalesian's didn't, which loops back to O.

Every result here feeds a concrete "what to finetune next" decision. Frame the handoff that way.

## Dependencies

P0's `eval_ordinal.py` for the metric block (stub it if P0 isn't merged yet — 1a/1b only need the
existing PLS path). Runs in parallel with all of Thrust B.

## What to read (repo)

- `README.md` §3 (activation loader `find_acts_dir`/`load_layer`, the maximal-balanced harness).
- `v_1/src/linear_probing/00_tokenization_check.py` — there is already a tokenization check; **extend it.**
- `v_1/src/linear_probing/03_extract_seal_activations.py` — the canonical extractor; it already
  handles the uMT5-style encoder (Thalesian is on disk), so it will extract **vanilla uMT5-base**
  the same way. This is what makes 1b nearly free.
- `v_1/src/finetune/eda/TOKENIZER_EDA.md` — why vocab expansion was skipped (relevant to factor T).
- `v_1/src/finetune/train_ntp.py` + `prepare_ntp_data.py` — the NTP harness with `--unfreeze-from`/`--lora`;
  **1c's objective arm is a sibling** — copy structure, swap the objective.
- `v_1/src/finetune/build_scoreboard.py` + `linear_probing/round2_phase0/run_mc_probes.py` — new
  methods auto-register from on-disk activations and tabulate into the same scoreboard.

## What to read (papers — see the separate paper review Yarin requested for plain-language notes)

- **Thalesian model card** (`Thalesian/cuneiformBase-400m`) — confirm uMT5-base + cuneiform tasks.
- **mT5 / uMT5** (Xue et al. 2021; Chung et al. 2023) — encoder-decoder, span-corruption, multilingual; grounds factors A and O.
- **Akkadian NMT / Akkademia (Gutherz et al. 2023)** — BLEU≈37 Akkadian→English; enables 1c's translation arm + 1d diagnostic.

## What to build (ordered by value-per-cost)

### 1a — Tokenization audit (CPU, instant — do immediately)
`v_1/src/chronorank/autopsy/tokenization_audit.py`. Same ORCC texts, per tokenizer
(Thalesian/uMT5, Qwen3, gpt-oss): tokens-per-Akkadian-word (fertility), UNK/fragmentation rate,
and handling of determinatives, diacritics, subscript numbers, logograms. Table + histogram.
**Isolates factor (T).**

### 1b — The control-ladder probe (1 GPU to extract vanilla uMT5 once, then CPU — THE cheap star)
No training. Extract **vanilla `google/umt5-base`** activations on ORCC (tier0+maximal, mean, all
layers) with the existing extractor, then probe it under the **identical** maximal-balanced PLS
alongside the already-on-disk models. The comparison ladder isolates each factor:

| Comparison | Holds constant | Isolates |
|---|---|---|
| Thalesian **vs** vanilla uMT5-base | tokenizer, architecture, objective-family | **(F)** the cuneiform finetune |
| vanilla uMT5-base **vs** Qwen3-8B base | (neither cuneiform-finetuned) | **(A)+(T)** enc-dec/bidirectional + tokenizer/pretraining bundle |
| 1a fertility numbers | — | **(T)** descriptively, to split (A) from (T) above |

Decision reads:
```
Thalesian ≈ vanilla uMT5         -> the WIN is the base model (arch/tokenizer/pretraining), NOT the finetune
Thalesian >> vanilla uMT5        -> the cuneiform finetune is doing the work  -> go to 1c (which objective?)
vanilla uMT5 >> Qwen base        -> encoder-decoder/bidirectional+multilingual base matters (factor A/T)
vanilla uMT5 ≈ Qwen base         -> base architecture is NOT it; the story is the cuneiform finetune (F/O)
```
This single experiment, with almost no compute, already narrows T/A/O/F a lot. **Run it first.**

### 1c — Objective ablation: seq2seq/translation vs NTP ⏸ ON HOLD (needs translation data)
*Do not build now.* When Yarin provides Akkadian→English translations: take Qwen3-1.7B or 8B and
finetune it two ways on the **same** data — the existing NTP arms (flat, on disk) **vs** a
seq2seq/translation objective — then probe both under maximal-balanced. `translation-LoRA > NTP-LoRA`
→ objective, not mere exposure, creates the date signal (the major result). Implement `train_seq2seq.py`
mirroring `train_ntp.py`; read translations via `--translation-path`/`--translation-col`.

### 1d — Date-from-English vs date-from-Akkadian diagnostic ⏸ ON HOLD (needs translation data)
*Do not build now.* When data lands: PLS-probe date from (a) Akkadian, (b) English, (c) both.
English dates well → topic/content confound; Akkadian >> English → language-internal signal.

## Cluster / sbatch (in `v_1/src/chronorank/autopsy/sbatch/`)

Only the two in-scope jobs for now:
- `P1a_tokenization.sbatch` — **CPU**, ~10 min, no `--gres`.
- `P1b_umt5_probe.sbatch` — `--gres=gpu:1` (extraction only), then CPU probe via `run_mc_probes.py`;
  mirror the extract→probe→scoreboard chain in `FT3_probe_qwen3_1b7_ft.sbatch`. **The priority run.**

Each ends with the commit+push pattern from `FT3_probe_qwen3_1b7_ft.sbatch`; print Yarin the one-line paste.
(`P1c_seq2seq_ablation.sbatch` / `P1d_english_diag.sbatch` — defer until translation data is provided.)

## Report back / success criterion

**PASS** when the factorial table is filled and you give a **signed conclusion on T/A/O/F** plus the
**downstream recommendation** ("therefore the next finetune should be …"). For the current scope,
**1a + 1b alone are the PASS** — they narrow T/A/O/F substantially with no training and no
translation data. (1c later confirms the objective story once translations arrive.) The single most
valuable sentence to produce: *"Thalesian wins because of **X**; therefore we should **Y** the big models."*
