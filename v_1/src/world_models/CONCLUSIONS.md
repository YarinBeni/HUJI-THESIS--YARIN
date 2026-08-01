# Conclusions — what the probes actually show

Everything below is read off committed result files. Every claim names the figure and
the table row it comes from, so it can be re-derived without cluster access. Companion
documents: `EXPERIMENT_MAP_MATRIX.md` (the design), `figures/README_READOUTS.md` (which
numbers the figures use and why), `../../../PRESENTATION_CONTEXT.md` (repo orientation).

---

## 0. The one result to lead with

**Linear time-decoding above a matched random-init control decays monotonically as the
stimulus gets more obscure and lower-resource, and it reaches zero at raw Akkadian.**

| cell | stimulus | best trained ρ | best random twin ρ | gap |
|------|----------|---------------|--------------------|-----|
| **A** | famous English names | Llama-2-70B **.921** | .667 | **+.254** |
| **B** | Assyrian ruler names, English | Llama-2-70B **.691** | .474 | **+.217** |
| **B′** | fragments, English gloss | AKK-300M **.422** | .281 | **+.141** |
| **C** | fragments, raw Akkadian | cuneiform-400M **.349** | .322 | **+.027** |

Read the **gap**, not the height. Absolute ρ is not comparable across rows — the
protocols differ (A is an i.i.d. entity hold-out, B is an entity-level Monte-Carlo,
B′/C are ruler-grouped MC over 8 rulers) — but "how far does this model get past its own
untrained twin, in the identical configuration" is comparable, and that is what decays.

Two things make this trustworthy rather than a story fitted to the data:

1. **It holds under both defensible read-outs.** Deck read-out: +.254 / +.217 / +.141 /
   +.027. Ridge everywhere: +.254 / +.211 / +.132 / +.007. Same shape.
2. **It does not hold under the leaky protocol** — the old table gives +.254 / +.211 /
   **+.079 / +.111**, which is non-monotone and has raw Akkadian *beating* the English
   gloss. The monotone ladder only appears once ruler leakage is removed. That is
   evidence the fix was real, not cosmetic.

I previously reported this ladder as "flat-then-cliff" off the leaky table. That
characterisation was wrong; the corrected numbers are a clean gradient.

Figures: `designs/slopegraph__deck.png` (the ladder directly), `00_MASTER_gap_by_stimulus_pooling.png`.

---

## 1. The protocol finding, which is a result in its own right

In the r8 fragment set, `year` takes **17 distinct values across 8 rulers**. The label is
very nearly the ruler's identity. So the cross-validation splitter is not a detail:

| splitter | a ruler is… | TF-IDF ρ on name-stripped Akkadian |
|----------|-------------|-----------------------------------|
| StratifiedKFold-by-ruler (`mc`) | in train **and** test | **.707** |
| GroupKFold-by-ruler (`mc_group`) | wholly train **or** test | **.330** ridge, **−.016** PLS |

A character n-gram model with no semantics reaches ρ .707 on text from which every
ruler's name has been stripped (verified: 0/344 fragments contain their own ruler's
transliterated spelling). It gets there by fingerprinting scribal orthography →
recovering ruler identity → reading the date off the identity. Under the grouped
splitter that route is closed and the baseline goes to zero.

**Consequences already applied:**

- Deck slides 25 and 27 described their own protocol as "stratified-by-ruler CV", which
  contradicted slide 2. Rebuilt from `mc_group` by
  `../stress_tests/results/rebuild_year_slides.py`.
- The English-gloss conclusion **inverted**. The old slide concluded "chronology here is
  surface n-gram statistics, not a learned timeline", because TF-IDF (.775) beat every
  model. Under the correct protocol TF-IDF is .066 and four trained arms are .36–.42,
  well clear of their random twins. The old conclusion was an artifact of the leak.
- **R² is no longer reported for the fragment cells.** A grouped test fold is
  essentially one ruler, so its year variance is ≈0 and R² degenerates to −0.22 for
  *every* arm, floor included. It carries no signal at this granularity; Spearman does.

---

## 2. Where the signal lives

**Entity level survives; document level does not.** A name token carries a usable time
coordinate even for obscure Assyrian rulers (cell B: .691, vs .474 for the random twin,
.288 for TF-IDF). A whole fragment does not — every arm collapses toward its control.

**Pooling matters more than model size at fragment level.** `mean` over the fragment
beats `last` for every arm without exception (e.g. Llama-2-70B on raw Akkadian: .311
mean vs .165 last). A document's final token is a poor summary of the document. The
deck's paper-faithful `last` site understates the fragment cells; both are now shown.

**Scale buys little once the stimulus is obscure.** At cell A the ladder is
monotone in size. At cell C the ordering is cuneiform-400M (.349) > Llama-2-70B-random
(.322) > Llama-2-70B (.311) — a 400M domain encoder beats a 70B general model, and an
*untrained* 70B sits between them.

Figures: `01_cross_levels.png`, `02–08` (layer sweeps and PLS curves per cell),
`designs/heatmap.png` (every cell × pooling × probe in one matrix),
`designs/dumbbell__deck.png` (trained-vs-twin per configuration).

---

## 3. Honest caveats — state these, do not bury them

- **The raw-Akkadian cell is weak, not clean, evidence.** Llama-2-70B-random scores .322,
  third overall. With 8 held-out rulers the trained/random margin is inside the noise for
  every arm except cuneiform-400M. This cell should be described as consistent with a
  weak learned signal, not as demonstrating one.
- **The probe choice changes who wins at fragment level.** Ridge: TF-IDF ranks **1/15**
  (.330). Best-k PLS: TF-IDF ranks **15/15** (−.016). Ridge on 256-dim TF-IDF-SVD still
  fits the residual orthographic signal; the low-rank PLS projection does not. The deck
  reports PLS, so deck-facing figures use PLS — but the divergence is real and is
  documented rather than hidden (`figures/README_READOUTS.md`).
- **PLS `k` was never fitted.** The grid `{1,2,3,5}` was inherited from
  `../stress_tests/shared/mc_probe.py`, and 18 of 58 fragment cells came back pinned at k=5 — the grid,
  not the data, was choosing k. `WAk_pls_ksweep` / `WBk_entity_ksweep` re-run with k
  spanning 1–64. **Numbers in this document predate that sweep and may move.**
- **`pls_best_k` is selected on the outer test folds**, which is what the deck does and
  is optimistic — more so as the grid widens. `mc_group` now also reports
  `pls_nested_spearman_mean`, with k chosen inside the training rulers. Quote the nested
  value once WAk lands.
- **n is small where it matters.** 8 rulers, ~1,070 fragments, 204 ruler-name rows.

---

## 4. Geometry — the manifold analysis

Two diagnostics from Modell et al. (arXiv 2505.18235) run on **full activations**, 332
configurations (`manifold/figs/`, 664 PNGs + per-run `stats.json`):

- **ρ** — Pearson(kNN-graph geodesic distance, feature distance). Behaves like a
  world-model measure: it separates trained from random. World places: **.290 vs .101**.
  Famous figures: .105 vs .024. Ruler names: .343.
- **ξ** — Chatterjee(squared feature distance, cosine similarity). Does **not** separate.
  At fragment level Llama-2-70B reaches .556 and its own random twin .463; an untrained
  Qwen on the English gloss reaches .531. ξ there is measuring the shape of the
  activation cloud, not learned chronology.

The practical upshot: **do not quote ξ as evidence of a world model.** Quote ρ, and
quote it against the matched control.

Figures: `19_isometry_summary.png` (the whole comparison in one panel — this is the one
for the deck), `17_arc__*.png` / `18_isometry__*.png` (single-cell detail),
`15_reducibility_indices.png`, `16_year_metric_choice.png`,
`09–14` (PLS and UMAP embedding six-panels; PLS forges a weak supervised axis at
fragment level where unsupervised UMAP shows none — the cleanest visual statement of the
entity→document cliff).

---

## 5. Figure inventory for the deck and the PDF

Every figure is regenerable; nothing is hand-edited.

| figure | what it says | rebuild |
|---|---|---|
| `designs/slopegraph__deck.png` | **the ladder** — Δρ vs matched control across A→B→B′→C | `designs/slopegraph.py` |
| `designs/heatmap.png` | every cell × pooling, ridge **and** PLS panels side by side | `designs/heatmap.py` |
| `designs/ridgeline__deck.png` | distribution of all 14 arms per configuration (discrete 0.05 bins) | `designs/ridgeline.py` |
| `designs/dumbbell__deck.png` | trained vs random twin, one row per arm × configuration | `designs/dumbbell.py` |
| `designs/anatomy__deck.png` | what the probe reads, and what it finds | `designs/anatomy.py` |
| `designs/geometry__deck.png` | PLS vs UMAP embeddings for the four cells | `designs/geometry.py` |
| `19_isometry_summary.png` | ρ separates trained from random; ξ does not | `manifold/make_isometry_summary.py` |
| `00`–`08` | master gap figure, layer sweeps, PLS curves | `figures/make_curves.py` |
| `09`–`16` | embedding six-panels, reducibility, metric choice | `figures/lib/`, `manifold/` |

Each of the six designs renders in three read-outs: `*.png` (ridge everywhere),
`*__deck.png` (the per-cell probe the thesis reports), `*__pls.png` (best-k PLS).
`heatmap` has no variants — it shows both probe panels by construction.

    cd figures && python3 build_tidy.py --mode mc_group --readout deck
    cd designs && TIDY_CSV=../TIDY_all_year_results__mc_group__deck.csv FIG_TAG=__deck python3 slopegraph.py

---

## 6. Open items

1. `WAk_pls_ksweep` and `WBk_entity_ksweep` have not been launched. Until they land,
   every PLS number here uses the truncated `{1,2,3,5}` grid and 18 fragment cells are
   pinned at its ceiling.
2. After they land: rebuild the tidy tables, re-render all three read-out sets, and
   switch the fragment quotes to `pls_nested_spearman_mean`.
3. The geo (space) cells are reported by-site with `StratifiedKFold-by-site`, which is
   the *seen-site* protocol — deliberately, as the space analog of the paper's setup.
   It is not the grouped protocol used for year, and the deck should say so where the
   two sit on adjacent slides.

---

## 7. Figure output format

Every figure is written by `figures/lib/_save.py` as a **300 dpi PNG** (for the HTML
deck) and a **vector PDF** (for the thesis). Use the PDF in LaTeX. Earlier renders were
120–220 dpi and are not usable in a talk or a paper; nothing at that resolution remains.

    FIG_DPI=450 python3 <script>.py    # poster resolution
