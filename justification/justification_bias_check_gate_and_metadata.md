# Justification — The bias-check gate (permutation test) and metadata-leak control

> **Thesis claim this supports:** "Before trusting any dating/period result we ran a formal
> bias-check: can a simple classifier recover the period from *transliteration alone*, and is
> that signal statistically real? We found (a) the signal is genuine (permutation p = 0.001),
> so the task is learnable, and (b) the easy 99% came largely from **metadata leakage**
> (`corpus_source` is a near-perfect period proxy), which is why we evaluate on cleaned text
> and treat metadata as a confound to control, not a feature to use."

## 1. The decision, in one sentence

We gated the whole probing program on a **permutation-tested bias check** and used its
diagnosis (the three-tier signal hierarchy + metadata leakage) to define what counts as a fair
evaluation — directly motivating the cleaning regime and the metadata controls.

## 2. The two findings that the rest of the thesis rests on

### 2.1 The chronological signal is real (not an artifact) — so probing is justified

The bias check trains 8 classifier variants (MLP depth 1–5 + Attention+MLP) on TF-IDF char
n-grams and assesses each with a **1,000-permutation test** (`v_1/src/bias_check/`, README).
On the SEAL multi-task version every metadata task is significant at **p = 0.001**:

| Task | tier0 F1 | maximal F1 | p |
|---|---|---|---|
| domain | 0.952 | 0.889 | 0.001 |
| period | 0.608 | 0.464 | 0.001 |
| genre | 0.361 | 0.269 | 0.001 |
| sub_genre | 0.286 | 0.267 | 0.001 |
| provenance | 0.171 | 0.128 | 0.001 |

(`v_1/src/bias_check/README.md` §"Phase C results"). This is a **positive** result: diachronic
structure is detectable from surface form, validating that there is something for the linear
probes to find. Without this gate, a probe's success would be unfalsifiable.

### 2.2 The 99% on letters was mostly metadata leakage — so we control metadata

The `bias_analysis.ipynb` diagnosis (README §"Analysis Notebooks") found the dominant driver of
the 99% letter-classification accuracy is **`corpus_source` acting as a perfect proxy for
period** (archibab→OB, oracc→NA, lbl→LB). Hand-crafted *linguistic* features combined reach
only **77.9%**, well below TF-IDF's 99% — i.e. the headline accuracy is inflated by a
provenance artifact, not by language alone.

The geodesic phase quantifies the same leak for the *dating* target: a regressor on **metadata
alone** scores Spearman **0.616** (imbalanced, `T8_metadata_baseline.csv`) — higher than *any
text model* — but collapses to **0.203 under the balanced protocol**. Two lessons feed forward:

1. **Never let metadata into the dating features** — it would dominate everything (this is why
   the probes read *text activations*, not corpus tags).
2. **Balancing alone removes most of the metadata advantage** (0.616→0.203), an independent
   argument for [[justification_balanced_mc_protocol]].

## 3. The three-tier signal hierarchy → defines `clean_maximal`

The same notebook decomposes the surface signal into three tiers (README §"Three-tier signal
hierarchy"):

1. **Writing conventions** (determinatives, subscript homophone digits, logogram rate) —
   *removable* by cleaning.
2. **Phonology/morphology** (syllable patterns, case endings) — *partially* removable.
3. **Content/pragmatics** (deity/place names, formulae) — *not* removable without destroying
   meaning.

The 11-step greedy cleaning ablation (99.2%→96.8% on 2–5-grams; 84.8%→69.3% on unigrams)
becomes the `clean_maximal` pipeline used everywhere downstream. Crucially, accuracy does *not*
fall to chance — "you can remove markup conventions but not Akkadian phonology" — which is the
evidence that a *real* linguistic signal survives the cleaning. This is the empirical
foundation of [[justification_maximal_cleaning_regime]].

## 4. Why TF-IDF char n-grams were the right bias-probe

(README §"Featurization Design") Zero learnable parameters (no overfitting on ~3.5k samples),
unambiguous interpretation of failure, and `char_wb` 2–5 grams capture exactly the
orthographic/morphological conventions a confound would live in.

## 5. Supporting literature

- **Ojala & Garriga (JMLR 2010), "Permutation Tests for Studying Classifier Performance."**
  Cited in the bias-check README as the standard that established **permutation testing with
  simple classifiers as the gold standard for dataset-bias checks**. This is the direct
  methodological authority for our gate. **[direct — add to `papers/` for the thesis bib.]**
- **Yoffe, Dershowitz, Vishne & Sober — "…Sequentially Correlated Literary Properties…"**
  (`papers/txt/Ancient Language papers/`). Same hypothesis-testing-against-a-null spirit; warns
  that neural/supervised models mistake structural confounds for the target — precisely the
  metadata-leak failure we caught. **[analogous/supporting.]**

## 6. Figures & tables to pull when writing

- Bias-check verdict + permutation null: `v_1/data/evaluation/bias_check/.../plots/permutation_test.png`,
  `bias_check_report.md`.
- Metadata-leak diagnosis: `v_1/src/bias_check/bias_analysis.ipynb` (§2 metadata, §9 summary,
  §10 cleaning ablation, §12 entity bias).
- Metadata-only dating baseline: `v_1/src/geodesic/results/tables/T8_metadata_baseline.csv`
  (0.616 imbalanced → 0.203 balanced).
- SEAL multi-task table: `v_1/src/bias_check/README.md` §"Phase C results".
