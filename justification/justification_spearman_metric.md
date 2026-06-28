# Justification — Spearman rank correlation as the headline dating metric

> **Thesis claim this supports:** "We report **Spearman rank correlation** between predicted
> and true year as the primary dating metric because (a) the historical question is ordinal —
> *did this text come before or after that one* — (b) Spearman is invariant to the
> monotonic/scale differences between models with different hidden dimensionalities and output
> calibrations, and (c) prior temporal-representation work scores readouts with rank/linear
> correlation."

## 1. The decision, in one sentence

The headline number is **Spearman ρ between predicted year and gold year** (with R², MAE,
MASE, MdAPE reported alongside), because chronology is fundamentally a *ranking* problem and
Spearman gives a confound-robust, cross-model-comparable score.

## 2. Why Spearman specifically (the three reasons)

### 2.1 The task is ordinal, not point-regression

What a historian wants is the **relative ordering** of texts in time; exact-year MAE is
secondary and noisy given ±decade label uncertainty. Spearman scores the ordering directly.
This is why the geodesic phase's geometric readout is evaluated with `pairwise_order_acc` /
Isomap rank-agreement (`v_1/src/geodesic/utils.py`) — the same ordinal philosophy, and
Spearman is its continuous summary.

### 2.2 It neutralises the cross-model dimensionality / calibration confound

This is the reason you flagged directly. We compare models with **very different hidden
dimensionalities and output scales** — TF-IDF vectors, a 37M MLM, Thalesian 300M/400M cuneiform
encoders, and Qwen3 1.7B→32B (hidden sizes from a few hundred to thousands). A metric tied to
absolute predicted-year values (raw MAE, or R² before centering) would conflate "reads
chronology well" with "happens to be calibrated on the same scale." **Spearman depends only on
the rank order of predictions**, so it is invariant to any monotonic re-scaling the PLS head
applies and lets all eight models sit on one comparable axis. (PLS itself is the other half of
this dimensionality control — see [[justification_pls_regression]].)

### 2.3 It is robust to the year-distribution skew

ORCC years are clumped by dynastic survival (see [[justification_balanced_mc_protocol]]).
Pearson/R² are dragged around by a few high-leverage outlier years; Spearman, being
rank-based, is not — which is why it stays interpretable under the imbalance we deliberately
do *not* fully erase.

## 3. How it is reported (so the metric can't flatter a model)

- Always as **mean ± std over the 200 MC balanced draws** (e.g. `0.3633 ± 0.0841`,
  `v_1/src/finetune/results/scoreboard.md`).
- Always next to its **shuffled-year null** (`shuffled_spearman_mean` in
  `T_headlines.csv` / `T1_year_pls.csv`) so "above chance" is explicit.
- Computed under **GroupKFold by ruler** so the rank correlation reflects generalisation
  across rulers, not name memorisation.

This is also the metric that carries the project's two load-bearing findings:

- **Length-confound finding:** TF-IDF Spearman collapses under truncation while Thalesian/Qwen
  hold (0.474→0.245 vs 0.430→0.407; [[justification_maximal_cleaning_regime]]).
- **Finetune null finding:** NTP finetuning does not move maximal Spearman at any scale (e.g.
  `qwen3_32b` base **0.3398** vs ft **0.3398**, `scoreboard.md`); the one apparent gain
  (gpt-oss-120b tier0 ft00 0.4038→0.4514) is a length confound that vanishes at maximal.

## 4. Supporting literature

- **Gurnee & Tegmark — "Language Models Represent Space and Time"**
  (`papers/txt/Geometric Representation papers/LANGUAGE MODELS REPRESENT SPACE AND TIME.txt`).
  The canonical precedent for *evaluating a temporal/spatial linear readout with correlation
  between probe output and ground-truth coordinate*. Our temporal-direction-plus-correlation
  setup follows Gurnee & Tegmark directly; we use Spearman (rank) rather than Pearson/R² for
  the cross-model invariance reasons in §2.2. **[direct — same probing-then-correlation
  evaluation; we ran it "following Gurnee & Tegmark," see
  `yarin/research_plan/planning/stream2_amir_connection.md:56`.]**
- **Yoffe, Dershowitz, Vishne & Sober — "Estimating the Influence of Sequentially Correlated
  Literary Properties…"** (`papers/txt/Ancient Language papers/`). Frames classification
  reliability via *rank/order-preserving surrogate statistics* and pairs the metric with a
  null distribution — exactly our Spearman-plus-shuffled-null reporting. **[supporting.]**
- **Geometry-of-representations papers** ("The geometry of hidden representations of large
  transformer models"; "The Geometry of Categorical and Hierarchical Concepts")
  (`papers/txt/Geometric Representation papers/`). Motivate reading a concept (here, time) off
  a low-dimensional linear subspace, whose *direction* is what a rank correlation cares about,
  not its absolute scale. **[supporting — justifies a scale-free readout metric.]**

## 5. Tables & figures to pull when writing

- Headline Spearman (tier0): `v_1/src/geodesic/results/tables/T_headlines.csv`.
- Maximal Spearman per layer: `v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv`.
- Spearman-vs-size & layerwise: `maximal_figs/figures/fig2_maximal_AB.png`, `fig4_maximal_A.png`;
  round-3 versions `results/figures/round3_story/fig2_model_size_scaling.png`,
  `fig4_layerwise_depth.png`.
- Finetune Spearman scoreboard: `v_1/src/finetune/results/scoreboard.md`,
  `figures/maximal_pls_bestlayer.png`.
