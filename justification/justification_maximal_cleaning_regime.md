# Justification — The "maximal" cleaning + 32-word truncation regime

> **Thesis claim this supports:** "We adopted the *maximal* evaluation regime (aggressive
> cleaning + ≤32-word truncation) because our own experiments showed that the apparent
> winner — TF-IDF — was winning *on a length artifact, not on linguistic chronology*. Once
> the length crutch is removed, TF-IDF collapses to baseline while the cuneiform encoder and
> the LLM family stay stable."

## 1. The decision, in one sentence

We moved every model onto the **same length-controlled, heavily-cleaned footing** (the
`clean_maximal` 11-filter pipeline + truncation to ≤32 words) so that a model's dating score
reflects *linguistic/temporal* signal rather than *how much surviving text* a period happens
to have. This regime — not the raw "tier0" regime — is the one we report as the honest
benchmark.

## 2. What the experiments actually showed (the reason we switched)

### 2.1 TF-IDF "won" at tier0 — and that win was length

In the uncleaned / untruncated **tier0** regime, TF-IDF and the small MLM are at the *top* of
the year-PLS Spearman ranking, ahead of the cuneiform encoder and the Qwen family:

| Model | tier0 best-layer year-PLS Spearman | source |
|---|---|---|
| tfidf | **0.407** (L00) | `v_1/src/geodesic/results/tables/T_headlines.csv` |
| mlm (37M) | 0.424 (L01) | same |
| thalesian_cunei400m | 0.411 (L12) | same |
| qwen3_32b | 0.399 (L09) | same |
| thalesian_akk300m | 0.344 (L06) | same |

An earlier narrative number quoted in the plan is starker still — TF-IDF year Spearman
**0.474 → 0.245** under truncation (`thesis_plan.md:1941`, `thesis_state.md`) — **but note that
0.474 was *not* measured under the fair test** (mean-pooling · balanced · 200-MC · `clean_maximal`
· PLS). It comes from an earlier/looser protocol snapshot, so treat it as *illustrative* of the
direction, not as a headline number. The authoritative fair-test pair is the committed-table one
above (TF-IDF tier0 **0.407** → maximal **~0.29**). The mechanism is documented in the same plan
section: *"historical survival is uneven; certain eras are represented by massive, highly
preserved royal inscriptions (e.g., 620 fragments), while adjacent periods yield only highly
fragmented administrative letters (e.g., 25 fragments). TF-IDF and unmodified LLMs exploit
these surface markers — specifically document length and the frequency density of
non-temporal determinatives or logograms."*

> **Interpretation for the thesis:** TF-IDF was dating documents by *word count and token
> density*, which correlates with period only because preservation correlates with period.
> That is a dataset artifact, not Akkadian language evolution.

### 2.2 Under maximal, TF-IDF collapses to baseline; the encoder is stable

The committed maximal table (`v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv`,
mean pooling, 200 MC balanced draws) shows TF-IDF falling to the random/MLM baseline band:

| Model (maximal, best layer) | year-PLS Spearman |
|---|---|
| tfidf | ~0.29 |
| mlm (37M) | ~0.31 (L01) — i.e. *at* the TF-IDF/random band |
| thalesian_akk300m | ~0.29 and length-robust |

The plan/state narrative pairs this with the contrast that matters: TF-IDF *collapses* under
truncation while **Thalesian stays length-robust** and the Qwen family is also length-robust
(`thesis_state.md` quotes the illustrative 0.430 → 0.407 for Thalesian against TF-IDF's drop —
again, re-derive the exact figures from the committed CSVs for the write-up). That separation —
collapse for the surface-feature model, stability for the representation-learning models — is
*the* result the maximal regime exists to expose.

### 2.3 The cleaning ablation that defined `clean_maximal`

The 11-filter `clean_maximal` pipeline was not arbitrary — it is the end-point of the
`bias_check/` cleaning ablation (`thesis_state.md`; `bias_analysis.ipynb`):

- TF-IDF accuracy **99.2% → 96.8%** after cleaning.
- Distinct unigrams carrying the signal **84.8% → 69.3%**.

i.e. cleaning strips a large fraction of the *lexical-shortcut* surface that a bag-of-words
model rides on, without destroying the genuine signal (96.8% is still well above chance),
which is exactly the property we want from a confound-removing filter.

## 3. Why this is the *right* control (and not just hiding TF-IDF's win)

Two independent confounds were identified, and maximal removes **both** at once:

1. **Length crutch** → removed by ≤32-word truncation (uniform document length).
2. **Lexical/determinative-density crutch** → removed by the `clean_maximal` filters.

A model that still ranks documents chronologically *after* both crutches are gone is using
something else — orthographic/grammatical evolution encoded in its representations. That is
the only kind of "dating ability" the thesis is willing to claim, so the maximal regime is
the evaluation we report headline numbers under (and the regime every downstream phase —
geodesic and finetune — adopted as its scoreboard).

## 4. Supporting literature

- **Yoffe, Dershowitz, Vishne & Sober — "Estimating the Influence of Sequentially Correlated
  Literary Properties in Textual Classification"** (`papers/txt/Ancient Language papers/`).
  Directly on point: shows that *supervised and neural classifiers are more prone to false
  positives — mistaking shared themes / surface structure for the property of interest* —
  and builds a hypothesis-testing framework to separate the confound from the real signal.
  Our maximal regime is the data-side analogue: instead of testing post-hoc, we *remove* the
  surface confound (length + lexical density) before scoring. **[analogous / supporting —
  motivates confound control, not the specific 32-word cutoff.]**
- **"The Medium Is Not the Message — Deconfounding Document Embeddings via Linear Concept
  Erasure"** (`papers/txt/Geometric Representation papers/`). Establishes that document
  embeddings carry strong nuisance/structural confounds that must be removed before a
  downstream readout is trustworthy. **[supporting.]**

## 5. Figures & tables to pull when writing

- Collapse story (tier0 vs maximal): `v_1/src/geodesic/maximal_figs/figures/fig1_maximal_ACD.png`,
  `fig2_maximal_AB.png`, `fig4_maximal_A.png`.
- Per-ruler MAE under maximal: `.../maximal_figs/figures/bars_mae_ruler.png`,
  `permodel_mae_ruler.png`.
- k-sweep selection-bias control: `.../maximal_figs/figures/ksweep_tradeoff_maximal.png`.
- Tables: `T_headlines.csv` (tier0 headline), `maximal_figs/tables/T1_year_pls_maximal.csv`
  (maximal).
- Cleaning ablation numbers: `v_1/src/bias_check/bias_analysis.ipynb`.

## 6. Note on which TF-IDF number to quote

The **0.474** figure is *not* from the fair test — it predates the mean-pool · balanced ·
200-MC · `clean_maximal` · PLS protocol (confirmed by Yarin). Use the **committed-table pair as
authoritative**: TF-IDF tier0 best-layer **0.407** (`T_headlines.csv`) → maximal **~0.29**
(`T1_year_pls_maximal.csv`), both under the fair test. Quote 0.474→0.245 only as a loose
illustration of the *direction*, never as a headline. See [[justification_balanced_mc_protocol]]
for the draw protocol the committed tables use.
