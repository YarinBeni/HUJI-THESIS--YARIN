# Justification — Probing internal activations instead of prompting the model

> **Thesis claim this supports:** "We read chronology off the model's *internal activations*
> (linear probes / PLS) rather than by *asking the model to date the text*. Our Round-2
> elicitation experiments showed the gap directly: the models hold the relevant facts (they can
> answer a king's dates ~88% of the time) but **cannot be prompted to date a fragment** — every
> prompt variant lands near chance — whereas a linear probe on the activations recovers the
> signal. The knowledge is *present but not elicitable*, so probing is the right instrument."

## 1. The decision, in one sentence

We use **activation probing**, not zero-shot prompting, as the dating readout, because our own
elicitation sweep (Round-2 phase 1a/1b) demonstrated that prompting fails even when the
underlying knowledge is provably in the model.

## 2. The two-sided evidence

### 2.1 The models *do* know the facts (knowledge probe, T9)

Direct knowledge-probing prompts (`round2_phase1a`, table
`v_1/src/geodesic/results/tables/T9_elicitation.csv`): asked for a *named king's* dates, the
models are accurate within ±50 yr most of the time —

| Model | kp0 accuracy (±50 yr) |
|---|---|
| qwen3_1b7 | 0.875 |
| qwen3_8b | 0.875 |
| qwen3_32b | 0.75 |

So the chronological knowledge is genuinely stored — this is *not* a "the model is ignorant"
situation.

### 2.2 But prompting to date a *fragment* fails (prompt re-probe, T10)

Four prompt variants (`round2_phase1b`, pv0–pv3, re-extracted and re-probed, table
`T10_prompt_reprobe.csv`): best ruler **macro-F1 ≈ 0.13–0.15** across all models and prompts,
against a chance level of **0.059** (1/17) — i.e. barely above chance and *nowhere near* the
~99% the linear probe reaches on the letter task. Prompt engineering does not move it: 32B is
flat at 0.133 across pv0–pv3.

> **Interpretation for the thesis:** there is a *representation–behaviour gap*. The temporal
> information is linearly present in the activations (probes recover it) but the model's
> generative interface cannot surface it for an unseen fragment. Measuring dating ability
> through prompting would therefore *understate* what the model encodes and confound "what it
> represents" with "what it can be made to say."

## 3. Why this makes probing the correct (and fairer) instrument

- **It measures representation, not instruction-following.** The thesis question is whether the
  *representation* contains a chronological coordinate (`thesis_plan.md:60, 112`), which is a
  probing question by construction.
- **It is model-fair.** Prompting advantages instruction-tuned chat models and penalises the
  encoder (Thalesian) and the MLM, which have no chat interface at all; a frozen-activation
  probe puts all eight models on the same readout. (See [[justification_spearman_metric]],
  [[justification_pls_regression]].)
- **It is consistent with the elicitation literature** the plan already leans on: the prompt
  baseline is kept as a *control* (`T9`/`T10`), and the headline result is the probe.

## 4. Supporting literature

- **Gurnee & Tegmark — "Language Models Represent Space and Time"**
  (`papers/txt/Geometric Representation papers/`). The whole premise — that space/time live as
  *linear directions in activations*, recovered by probes rather than by asking the model — is
  the methodological backbone here. Our split between "facts are present" and "behaviour can't
  surface them" is exactly why one probes the residual stream. **[direct.]**
- **Representation-Engineering surveys** ("Representation Engineering for LLMs — Survey…",
  "Taxonomy, Opportunities, and Challenges of Representation Engineering")
  (`papers/txt/Geometric Representation papers/`). Establish reading/steering *internal
  representations* as a first-class alternative to prompting for probing model knowledge.
  **[supporting.]**

## 5. Tables & figures to pull when writing

- Knowledge present: `v_1/src/geodesic/results/tables/T9_elicitation.csv` (kp0 ±50 yr).
- Prompting fails: `v_1/src/geodesic/results/tables/T10_prompt_reprobe.csv` (ruler macro-F1 ≈
  0.13 vs chance 0.059).
- Prompt-elicitation figure: `v_1/src/geodesic/results/figures/round3_story/fig5_prompt_elicitation.png`.
- Probe success it is contrasted against: letters 99.1% (`thesis_state.md`), ORCC probe tables
  in `v_1/src/linear_probing/results/`.
