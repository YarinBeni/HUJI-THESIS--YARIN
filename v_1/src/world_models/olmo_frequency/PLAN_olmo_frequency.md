# PLAN: OLMo arm + training-data frequency dose-response

Advisor request (Gabi, 2026-08). Goal: turn the deck's binary salient/obscure
contrast into a continuous curve. For a model whose full pretraining corpus is
open and searchable, correlate, per entity, how often the entity appears in the
training data with how well a linear probe places that entity on the timeline.

This needs a model whose training data we can actually count over, which is why
OLMo: open weights AND open corpus. Counting frequencies in some other corpus
and probing a model trained on a different one breaks the causal link, so the
model arm and the counted corpus must be a matched pair.

## Links and tools

- OLMo project: https://allenai.org/olmo
- OLMo code: https://github.com/allenai/OLMo
- Checkpoints (HF): `allenai/OLMo-2-1124-7B` (primary), optionally
  `allenai/OLMo-2-1124-13B`. Needs a recent `transformers` (>= 4.47); verify
  the cluster env loads it before submitting.
- Training corpus: OLMo 2 pretrains on the olmo-mix / Dolma family; the exact
  mix per checkpoint is on each model card. HF: `allenai/olmo-mix-1124`,
  `allenai/dolma`. https://github.com/allenai/dolma
- WIMBD (What's In My Big Data), the advisor's link, search + counts over big
  corpora: https://wimbd.apps.allenai.org/ and
  https://github.com/allenai/wimbd (paper: arXiv 2310.20707)
- infini-gram, the practical programmatic route: exact n-gram counts over
  Dolma-class corpora through a free web API, no local index needed:
  https://infini-gram.io/ (paper: arXiv 2401.17377). Check which of its
  indexes matches the chosen OLMo checkpoint's training mix and say in the
  results note which index was used. Use WIMBD's app for spot checks; use
  infini-gram for the batch counts.

## What already exists in the repo (reuse, do not rebuild)

- Model registry: `v_1/src/world_models/wm_lib/registry.py`. Add the OLMo
  arm(s) plus a matched random-init twin, following how the Qwen/Llama arms
  and their `*_random` twins are declared.
- Cell A (English, salient entities): the world_models English line
  (extraction + `probe_layers_pls` style probing over the paper's datasets,
  `historical_figure` etc.). Run OLMo through it unchanged: same layers, same
  RidgeCV/PLS settings, same seeds.
- Cell B (English, obscure entities): the WB entity pipeline under
  `v_1/src/world_models/akkadian/`: `build_entity_datasets.py`,
  `extract_entity.py` (sites `ent_last`, `ent_mean`, `last`, `mean`),
  `probe_entity.py` (entity-level MC, 200 draws, 20% of entities held out),
  `aggregate_entity.py`, sbatch templates `sbatch/WB*.sbatch`. Run OLMo as a
  new arm, nothing else changes.
- Figure house style: import `COLORS`, `LABEL`, `ORDER`, `IS_CTRL` from
  `v_1/src/world_models/plot_cellA_figs.py`. Give OLMo a NEW family hue (teal
  is free, e.g. 7B `#14b8a6`, 13B `#0f766e`), its random twin joins the
  purple/dashed control convention. Add the arm to `ORDER` and `LABEL` there
  so every existing figure script picks it up on rerun.
- Cluster conventions: sbatch jobs source `_common.sh`; results are committed
  and pushed by the job itself (`sync_main` + `commit_push`).

## Tasks

1. **Arm registration.** Add `olmo2_7b` (+ `olmo2_7b_random` twin; 13B
   optional) to the registry. Verify tokenizer/offset-mapping behavior works
   with `extract_entity.py`'s span logic (it has a prefix fallback; test on a
   handful of ruler names first).
2. **Probing runs.** Cell A English line and Cell B entity line for the new
   arms, identical protocols to the existing arms. Do NOT invent new
   protocols; the value of the arm is comparability.
3. **Frequency counts.** One script, `count_frequencies.py`, in this folder:
   - Inputs: the cell A entity names (all ~4k historical figures, plus the
     other five datasets if cheap) and the cell B names (34 rulers, 25
     find-spots; count BOTH the ancient and modern name forms that
     `build_entity_datasets.py` uses, keep them as separate rows).
   - Query infini-gram for exact-string counts against the index matching the
     checkpoint's training mix. Rate-limit politely, cache to CSV.
   - Output: `results/entity_counts.csv` with columns
     `entity_type, entity, name_form, count, index_used`.
   - Sanity: hand-check ~10 entities against the WIMBD app; flag names that
     are common words (e.g. short or ambiguous strings) with a `noisy_count`
     column, since raw string counts overcount those.
4. **Dose-response analysis.** `analyze_frequency.py`:
   - Per entity, get a per-entity error from the probe outputs (cell A: error
     of the held-out prediction; cell B: mean absolute error of that entity
     across the MC draws in which it was held out; `probe_entity.py` already
     works entity-level, extend it to dump per-entity errors if it does not
     already).
   - Main read-out: Spearman correlation between log10(count+1) and
     per-entity error, OLMo trained arm vs its random twin (the twin is the
     control: if the correlation is about learned representation, the twin
     should not show it).
   - Also report the curve binned by frequency decade, and the cell B rulers
     overlaid on the cell A curve as the low-frequency end.
   - Confound to address explicitly: entity age correlates with rarity AND
     with dating difficulty. Report the correlation within century bins (or
     partial correlation controlling for year) alongside the raw one.
5. **Figure.** `plot_frequency_fig.py`, house style: x = log frequency in
   training data, y = per-entity probe error (or per-bin mean), OLMo solid,
   random twin dashed purple, one panel per target if both year and place are
   run. Output to `results/figs/fig_frequency_doseresponse.png`.
6. **Results note.** Short `RESULTS.md` in this folder: which checkpoint,
   which index, the correlation numbers (raw and age-controlled), and one
   sentence on whether frequency predicts success. This feeds a future slide;
   do not edit the deck.

## Acceptance checklist

- [ ] OLMo arm reproduces the qualitative cell A result (beats its twin and
      TF-IDF on salient entities) before any frequency claims are made.
- [ ] Counted corpus matches the probed checkpoint's training mix, stated in
      RESULTS.md.
- [ ] Correlation reported for trained arm AND random twin, raw AND
      age-controlled.
- [ ] All new results committed via the standard sbatch commit flow.
