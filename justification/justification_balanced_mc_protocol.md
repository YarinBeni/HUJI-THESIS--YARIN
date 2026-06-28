# Justification — "balanced" evaluation via Monte-Carlo ruler balancing (200 MC draws)

> **Thesis claim this supports:** "We score dating under a *class-balanced* protocol — equal
> representation across rulers, resampled over many Monte-Carlo draws — because the ORCC
> ruler/year distribution is severely imbalanced, and an imbalanced score rewards a model for
> predicting the majority period rather than for reading chronology."

## 1. The decision, in one sentence

Instead of evaluating on the raw, imbalanced ORCC ruler distribution, we **balance across a
fixed set of rulers (8) and average the metric over 200 Monte-Carlo class-balanced draws**
(`pls__mc_balanced_maximal`). This is the "imbalance crutch removal" that pairs with the
length-crutch removal in [[justification_maximal_cleaning_regime]].

## 2. Why imbalance had to be removed (the experimental motivation)

### 2.1 The ORCC distribution is extreme

ORCC chronology labels are derived from kings' names, and dynastic survival is wildly uneven:
some periods are represented by **~620 fragments** and adjacent ones by **~25**
(`yarin/research_plan/planning/thesis_plan.md:1941`). The ruler label is a **38-class**
variable (`linear_probing/05_compute_cls.py`), with a long tail of rulers carrying only a
handful of fragments.

### 2.2 What that does to an unbalanced score

On a raw distribution, a degenerate "always predict the dominant era" model scores well above
chance — the metric rewards exploiting the prior, not reading the text. This is the same
failure mode that produced the **Round-1 ORCC surprise**: layer-wise linear probes that hit
**99.1% on letters** *failed* on ORCC (`thesis_state.md`, Phase 4). Diagnosing that failure
(Round 2) is what drove the move to a balanced protocol.

### 2.3 The fix: fixed-ruler balancing + Monte-Carlo averaging

- **Balance to 8 rulers** so every class contributes equally → the score can't be inflated by
  the majority period.
- **200 MC draws**: because balancing throws away data, any single balanced subsample is
  high-variance; we resample the balanced sets 200× and report mean ± std. Every committed
  headline number carries this `± std` (e.g. `qwen3_8b` maximal **0.3633 ± 0.0841**,
  `v_1/src/finetune/results/scoreboard.md`), which is *only* meaningful because of the MC
  averaging.

## 3. The other half: leakage control by ruler grouping

Balancing across rulers is paired with **GroupKFold by ruler** in the year-PLS regression
(`linear_probing/05_compute_pls.py`; `thesis_plan.md:1364, 2728`). A ruler never appears in
both train and test, so the regressor cannot memorise "this king ⇒ this year." Combined with
the `name_masking.py` ablation (zeroing ruler-name tokens), this separates *chronology* from
*named-entity lookup* — the confound the balanced protocol is most exposed to. See
[[justification_pls_regression]] and [[justification_spearman_metric]].

## 4. Supporting literature (and an honest note on provenance)

- **Yoffe, Dershowitz, Vishne & Sober — "Estimating the Influence of Sequentially Correlated
  Literary Properties in Textual Classification: A Data-Centric Hypothesis-Testing
  Approach"** (`papers/txt/Ancient Language papers/`). This is the closest methodological
  precedent in our library: it *models label sequences as stochastic processes and generates
  surrogate labelings* (a Monte-Carlo resampling of labels) to test whether a classifier's
  success is real or a confound. Our 200 MC balanced draws are the same idea applied to class
  balance: resample the label assignment many times and report the distribution of the metric
  rather than one lucky split. **[analogous / supporting — same Monte-Carlo-surrogate logic.]**
- **Permutation / shuffled-label controls in our own pipeline.** Every headline row also
  carries `shuffled_spearman_mean` / `shuffled_r2_mean` columns
  (`T_headlines.csv`, `T1_year_pls.csv`) — the date-shuffled null. This is the same
  hypothesis-testing spirit and is the cleanest in-thesis citation for "the balanced score is
  above its own null."

- **Nathan's preprint (title/citation TBD).** Yarin's recollection is that the MC-balancing /
  imbalance-fighting design follows a **Nathan preprint** that is not yet in `papers/`. Cite it
  as "Nathan (preprint)" for now; **add the full title + author + venue once it is public.**
  **[primary — placeholder citation, to be completed.]**

> **Provenance note — keep two things separate.** Nathan **Wasserman** (with Streck) is also the
> author of **SEAL — Sources of Early Akkadian Literature**, the provenance of our 384-fragment
> SEAL corpus (`v_1/src/corpus/02_build_seal_corpus.py`); that is the *data-source* citation. The
> *methods* citation for MC balancing is the Nathan preprint above (placeholder until named),
> reinforced by the Yoffe/Sober surrogate-labeling paper and our own shuffled-label nulls. Don't
> conflate the SEAL data citation with the balancing-method citation in the write-up.

## 5. Figures & tables to pull when writing

- Balanced headline numbers with MC std: `v_1/src/finetune/results/scoreboard.md`,
  `v_1/src/geodesic/results/tables/T_headlines.csv`.
- Maximal balanced year-PLS per layer: `maximal_figs/tables/T1_year_pls_maximal.csv`.
- Ruler-classification / balancing tables: `T3_ruler_classification.csv`, `T3b_ruler_plsda.csv`,
  `T5_loro.csv` / `T5_loro_per_ruler.csv` (leave-one-ruler-out).
- Name-masking control: `T7_name_masking.csv` and panel D of
  `maximal_figs/figures/fig1_maximal_ACD.png`.
