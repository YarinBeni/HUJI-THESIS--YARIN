# Pillar 5 — SAE / sparse feature archaeology  ⏸ DEFERRED TO END

> **STATUS: DEFERRED (Yarin's call, 2026-06-13). Do not start until P1 has landed.**
> Interpretability is genuinely interesting but it should be **guided** by results, not run blind.
> The plan is: finish **P1** (the autopsy) and the P2/P3 honest-dating system first; if P1's
> findings lead us to a **second, lesson-informed finetune** of the big models (e.g. a
> translation/seq2seq arm), then doing SAE feature archaeology *afterwards* — on a model we
> actually understand and chose deliberately — is far more valuable than decomposing the current
> frozen activations now. So this pillar waits for (a) P1's signed conclusion and (b) ideally that
> second finetune. The existing 545-line plan at `v_1/src/sae/plan/PLAN.md` stays as-is until then.
>
> The full design is preserved below for when we un-defer it.

---

> **Agent brief (FROZEN — only when un-deferred).** SAE is for discovery/explanation, not
> performance. Read `README.md` first. **Requires a working head (P2/P3) and P1's findings.**
> Extend `v_1/src/sae/plan/PLAN.md`, don't restart.

## Goal

Decompose the learned chronological axis into human-readable features, then **causally validate**
them. For each top feature: correlation with date, stability across ruler splits, top-activating
texts, token/span triggers, deletion effect, confound score. Classify features into:
`real linguistic change | orthographic/sign convention | formulaic register | genre artifact |
ruler/name leakage | corpus artifact | unknown candidate marker`.

## Dependencies

**P2 (required):** a trained ChronoHead + its learned date direction `w`. **P3 (preferred):** the
shortcut-resistant head, so the features you mine are the honest ones. Reuses the existing
`v_1/src/sae/` scaffold. **1 GPU** for SAE feature extraction.

## What to read (repo)

- `v_1/src/sae/plan/PLAN.md` — the existing 5-analysis plan (sparse probe accuracy curve, probe-
  direction decomposition, per-period profiles, cross-layer, bigram differential) targeting the
  Arditi 131k SAE on Qwen2.5-7B at layers 7/15/23. **This pillar = execute + extend that plan to
  the ChronoHead direction and to Thalesian.**
- `v_1/src/sae/__pycache__/` shows `01_extract_sae_features.py`, `02_analyze.py`, `utils.py` once
  existed — recover/recreate from the plan if the `.py` sources are missing.
- P2's `model.py` (`sparse_linear` head variant) and the learned `w` you'll decompose.
- `v_1/src/chronorank/eval_ordinal.py` (P0) for the deletion-effect re-scoring.

## What to read (papers)

- **Anthropic dictionary-learning / "Towards Monosemanticity" / Scaling Monosemanticity** — features
  as better units than neurons; how to read top-activating examples and do feature ablation.
- **Arditi (2024) Qwen2.5-7B SAE (131k)** — the pretrained SAE referenced in the existing plan.
- Skim the plan's "Temporal SAE" idea from `thesis_plan.md` Plan 2 (§T-SAE) **only as an optional
  extension** — do not build the contrastive T-SAE first; standard SAE + the ChronoHead direction
  decomposition is the committed scope.

## What to build (on top of the existing plan)

1. **Probe-direction decomposition:** project the ChronoHead's learned date direction `w` onto SAE
   decoder columns → ranked list of features that compose "time". (Plan Analysis 2, applied to `w`.)
2. **Per-feature dossier** (`sae/feature_dossier.py`): for each top-k feature emit correlation-with-date,
   ruler-split stability (bootstrap), top-20 activating ORCC texts, token/span triggers (differential
   bigram analysis from the plan), and a **confound score** (can ruler/length/genre/corpus explain it?).
3. **Causal deletion test** (`sae/causal_deletion.py`): zero the feature (or mask its trigger spans),
   re-run P0 `eval_ordinal` on the head, record prediction shift vs a random-matched-span control.
   This is the difference between "correlated with date" and "causally used."
4. **Feature taxonomy table:** classify each top feature into the 7 buckets above.

## Cluster / sbatch

- `P5a_extract_sae.sbatch` — `--gres=gpu:1`, `--mem=128G`. Run SAE over the chosen layer(s)
  (Thalesian best layer + Qwen3-8B L16) on ORCC; cache feature activations (gitignore large arrays).
- `P5b_analyze.sbatch` — **CPU**, runs decomposition + dossiers + causal deletion + taxonomy,
  commits JSON + figures.

Give Yarin both paste commands; P5a before P5b. Follow the commit/push pattern.

## Report back / success criterion

**PASS** when: (a) the date direction is decomposed into a ranked, named feature list; (b) at least
a handful of features pass the **causal deletion** test (zeroing them moves the prediction in the
expected direction more than random spans); (c) the taxonomy table is filled, explicitly separating
*real linguistic/orthographic drift* from *ruler/genre/corpus leakage*. The headline is: "the model
moves this text later because features F_a, F_b fire; F_a = late orthographic pattern (causal),
F_c rejected as ruler leakage." That sentence, backed by numbers, is the Assyriological contribution.
