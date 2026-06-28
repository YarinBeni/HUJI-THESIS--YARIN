# Justification — Partial Least Squares (PLS) as the dating readout head

> **Thesis claim this supports:** "We read the year off frozen representations with **PLS
> regression** because (a) the activations are high-dimensional and collinear, with hidden
> sizes that *differ across models*, so we need a supervised low-rank projection that maps any
> model onto a common, comparable temporal subspace; (b) PLS finds the directions of the
> representation that *covary with year*, which is exactly the confound-aware, year-supervised
> projection we want; and (c) low-dimensional linear concept subspaces are the standard way
> recent work reads structured attributes out of LLM activations."

## 1. The decision, in one sentence

The dating head is **PLS regression (small k components) from frozen layer activations to
year**, chosen as the confound-resistant, dimensionality-normalising linear readout that lets
TF-IDF, the MLM, Thalesian, and the Qwen family be compared on one footing.

## 2. Why PLS specifically (the three reasons)

### 2.1 It normalises away the cross-model dimensionality confound

This is the concern you raised: different models have **different activation dimensionalities**
(TF-IDF sparse vectors; 37M MLM; Thalesian 300M/400M; Qwen3 1.7B→32B). A plain ridge/OLS
readout in each model's native space gives scores that aren't comparable and that reward big
models for having more raw dimensions to overfit. **PLS projects every model down to the same
small number of latent components (k≈3) before regressing** — so "how much temporal signal is
linearly accessible" is measured in a common k-dim space, not in each model's idiosyncratic
hidden size. (Spearman then makes the *output* scale-free too — see
[[justification_spearman_metric]].)

### 2.2 It is the *supervised, year-aware* projection — not generic variance

PCA would pick the directions of largest variance, which in these activations are dominated by
length/genre/lexical nuisance (the very confounds we fight in
[[justification_maximal_cleaning_regime]]). **PLS instead picks the directions of maximal
covariance with the year label**, so the components it keeps are the ones that *track
chronology*, and the nuisance variance is pushed out of the low-rank readout. This is the
direct, built-in way PLS "takes the years into account."

### 2.3 Collinearity + small n

The labeled set is small (~893 of 1,202 ORCC) and the activations are highly collinear; OLS is
ill-posed and overfits. PLS (like ridge) is regularised by the rank truncation, which is why it
is the standard chemometrics tool for *p ≫ n, collinear predictors, continuous target* — our
exact regime.

## 3. Guardrails we put around PLS (so the readout is honest)

- **GroupKFold by ruler** (`v_1/src/linear_probing/05_compute_pls.py`) → no ruler in both
  train and test; the PLS head cannot memorise king⇒year.
- **Fixed k = 3 for the headline MAE**, *not* best-k-per-draw. The maximal_figs README
  documents that best-k-per-draw inflates the score via selection bias (each draw post-hoc
  picks its luckiest k), so we fix k to kill that bias
  (`v_1/src/geodesic/maximal_figs/README.md`, "Open decision baked in"). The k-sweep itself is
  reported as a control (`ksweep_tradeoff_maximal.png`).
- **Name-masking + shuffled-year null** run on the same PLS pipeline (`name_masking.py`,
  `T7_name_masking.csv`, `shuffled_spearman_mean` columns) to show the PLS signal isn't ruler
  leakage and is above its null.

## 4. Why PLS and not the fancier alternatives (yet)

The plan is explicit that **PLS is the deliberately-simple baseline** the rest of the program
is measured against: *"Your current problem is not lack of a differential equation. It is
confounding, data sparsity, and lack of an objective that directly encodes chronology"*
(`thesis_plan.md:382`); *"frozen representation → PLS regression … ChronoRank-SAE adds four
things PLS does not have"* (`thesis_plan.md:949–953`). So PLS is justified as the **honest,
interpretable, low-variance floor**: anything more complex must beat PLS under the same
maximal-balanced protocol to earn its place. (To date, NTP finetuning does **not** beat it —
`scoreboard.md`.)

## 5. Supporting literature

- **Geometry-of-representations papers** — "The geometry of hidden representations of large
  transformer models," "The Geometry of Categorical and Hierarchical Concepts in LLMs," "The
  Hidden Lattice Geometry of LLMs" (`papers/txt/Geometric Representation papers/`). These
  establish that structured attributes live on **low-dimensional linear subspaces** of the
  activation space, recovered by *mean-difference / logistic probing / linear projection*
  (the Hidden-Lattice paper explicitly lists these). PLS is precisely a supervised
  linear-subspace estimator for a continuous attribute (year). **[direct — justifies a
  linear low-rank temporal subspace readout.]**
- **Gurnee & Tegmark — "Language Models Represent Space and Time."** Read space/time off
  activations with **linear probes**; our PLS is the continuous, collinearity-robust, low-rank
  variant of the same linear-probe idea. **[direct — same linear-readout family.]**
- **"The Medium Is Not the Message — Deconfounding Document Embeddings via Linear Concept
  Erasure"** (`papers/txt/Geometric Representation papers/`). Operates in the same space of
  *linear* operations on embeddings to separate a target concept from confounds — same toolkit,
  complementary direction (erase vs. project-onto). **[supporting — the confound-aware-linear
  rationale.]**
- **Yoffe/Sober surrogate-labeling paper** (`papers/txt/Ancient Language papers/`) for the
  null/surrogate pairing we wrap around the PLS output. **[supporting.]**

## 6. Tables & figures to pull when writing

- PLS year tables: `T1_year_pls.csv` (tier0), `maximal_figs/tables/T1_year_pls_maximal.csv`
  (maximal); ridge comparison `T2_year_ridge*.csv` (PLS-vs-Ridge is panel A of
  `maximal_figs/figures/fig1_maximal_ACD.png` / `fig2_maximal_AB.png`).
- k-sweep / selection-bias control: `maximal_figs/figures/ksweep_tradeoff_maximal.png`,
  `ksweep_per_method_maximal.png`, table `ksweep_tradeoff_maximal.csv`.
- PLS-DA ruler surface: `T3b_ruler_plsda.csv`, panel C of `fig1_maximal_ACD.png`.
- Code: `v_1/src/linear_probing/05_compute_pls.py`, `v_1/src/geodesic/maximal_figs/regen_figs_from_csv.py`.
