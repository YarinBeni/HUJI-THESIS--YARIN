# PLAN — Learning Akkadian time representations at scale (SSL + scaling sweep)

*Written 2026-09-02 after the E-MIN v2 / P1 results. Simple language on purpose.*
*Scaling-law sources requested (Weng 2026; Wolfe) could not be fetched from the*
*cluster network; the numbers below are from Kaplan et al. 2020, Hoffmann et al.*
*2022 (Chinchilla) and Muennighoff et al. 2023 (data-constrained scaling).*

## 0. The idea in three lines

1. **Pretrain without dates.** Make several views of every Akkadian text we have
   (masked names, crops, stripped formulas, normalised spelling) and train a
   model so that the views of one text get the same embedding (Barlow / JEPA).
   Dates are not needed for this — so we can use *all* ~40k texts, not 1,193.
2. **Fine-tune with dates.** Add the ordering loss on the 1,193 dated texts.
3. **Check the embeddings are meaningful before trusting the dates.** Texts
   with a known *period* or *genre* (thousands of them) must separate in the
   embedding space, and a probe must read their period.

## 1. What data we have (census, 2026-09-02)

| corpus (on disk) | texts | words | labels we can use |
|---|---|---|---|
| **Unified Akkadian corpus** (`v_1/data/unified/`) — ORACC 14,210 + eBL 24,909 + Archibab 1,310 | **40,429** | 2.45 M (4.9 M signs) | source, find-spot, composition place; genre for part |
| ORACC 1st-millennium letters/legal/admin (`corpus_b_oracc_1st_mill`) | ~2,500 | 273 k | **period** (6 classes), genre (3) |
| Letters, 3 period groups (`unified_3groups_akkadian_letters`, `texts_for_evaluation`) | 4,957 | 279 k | **period** (Neo-Assyrian / Old Babylonian / Late Babylonian) |
| SEAL literary (`seal_corpus`) | 384 | small | **period** (10), genre (16), sub-genre |
| ORCC royal inscriptions (`orcc_corpus`) | 1,202 | ~120 k | **year** (1,193), ruler, period, object type, find-spot |
| CDLI/ORACC metadata join (`oracc_cdli_metadata`) | 11,059 rows | — | period, genre, provenience for ORACC ids |

Facts that shape the plan:
* eBL texts are short (median 14 words); ORACC median 28; Archibab 38. Many
  fragments are tiny — a **minimum-length filter** (≥ 8 words) is needed.
* `value_signs` (sign-level tokens) exists for **all** sources and is consistent;
  `value_clean` is missing for Archibab. Sign level is the common representation.
* ~29 texts are duplicated between eBL and ORACC by content; 136 ORCC documents
  share text (exemplars). Deduplicate by content hash before any split.
* Zero year labels outside ORCC → **period** is the label we can probe at scale.

**Sources to add** (in order of value/effort): full ORACC (all projects; we hold
only a subset), CDLI ATF transliterations (largest catalogue), the rest of eBL,
Archibab full export. Each needs one normalisation pass to sign-level tokens.

## 2. The scaling reality — read this before sizing anything

* Total text ≈ **5 M sign tokens**. Chinchilla's compute-optimal rule is ~20
  tokens per parameter → a **from-scratch** compute-optimal model is ~**250 k
  parameters** for one pass. Data-constrained scaling (Muennighoff 2023): repeating
  data up to ~4 epochs costs almost nothing, value decays by ~16 epochs, and
  beyond ~40 epochs extra repeats are worthless. So the *useful* budget is
  ~20–80 M effective tokens → **1–4 M parameters** from scratch. A "big model
  from scratch" is not the path here; **we are data-constrained, not
  compute-constrained.**
* Therefore the "big" axis is **pretrained encoders** (cuneiformBase-400m,
  AKK_300m, Llama-2-7B/13B/70B, Qwen3-8B/32B) with **continued SSL training of
  an adapter**, plus a **small from-scratch family** (0.25 / 1 / 4 / 16 M
  params) to draw a real scaling curve and to see where from-scratch meets
  frozen-pretrained.
* Adapter/head sizes: 0.3 / 1 / 3 / 10 M parameters (we ran only 0.85–2.4 M).
* Fine-tune data fractions: 25 / 50 / 100 % of the 1,193 dated texts (× 5 seeds),
  to see whether the SSL pretraining reduces how many labels we need — that is
  the practical claim of the method.

## 3. Architecture family to compare

| name | what | why |
|---|---|---|
| **Barlow twin** (current) | two views → same embedding; EMA target branch; redundancy reduction | what we have; strong baseline |
| **BYOL-style** | add a predictor MLP on the online branch | stabler with small batches |
| **JEPA** | mask a span of the text; predict the *embedding* of the masked span from the context embedding (in latent space, no reconstruction) | natural for fragmentary tablets (lacunae); the objective the plan was named after |
| **Contrastive (InfoNCE)** | pull views together, push other texts apart | the standard comparator |
| **Supervised-only** | ordering loss alone on the 1,193 | control for "does pretraining help at all" |

All share the same views, tokeniser (sign level), encoder sizes, and read-out.

## 4. Avoiding ORACC bias

1. **Balanced sampling**: sample sources with probability ∝ nᵅ, α = 0.5 (temperature
   sampling), so eBL/ORACC do not swamp Archibab.
2. **One representation**: everything in sign-level tokens from `value_signs`.
3. **Source is a nuisance label**: a probe for *source* on the embeddings is
   reported next to the period probe. If source is easy and period is not, the
   model learned corpora, not time.
4. **Held-out source**: one evaluation with a whole source held out (e.g. train on
   ORACC+eBL, probe period on Archibab).
5. Period labels are correlated with source (eBL ≈ 1st-millennium literary,
   Archibab ≈ Old Babylonian). Report period probes **within source** as well.

## 5. How we will know the embeddings are meaningful (no dates needed)

* **Period probe** (linear and MLP, balanced accuracy, ruler/tablet-grouped folds)
  on: the 4,957 letters (3 periods), ORACC 1st-mill (6 periods), SEAL (10 periods).
* **Genre / provenance probes** as secondary structure checks.
* **Separation**: UMAP of embeddings coloured by period; **silhouette score** per
  label with a **permutation null** (shuffle labels 200×) so "looks separated"
  becomes a number. This is the quantitative version of the GUI we had.
* **Retrieval**: k-NN period purity (does a text's nearest neighbours share its period?).
* **Dated benchmark**: the 40-king ordering (mc ρ, pooled ρ, block null) as now.
* **Bias indicator**: the source probe (section 4).

## 6. Phases, deliverables, gates

| phase | work | GPU | deliverable / gate |
|---|---|---|---|
| **S0 data** | unify all corpora to one sign-level table; dedupe; min-length; source-balanced splits by tablet; period/genre label table | 0 | `chrono/artifacts_ssl/corpus_all.parquet` + census report; gate: no split leakage, dedupe count logged |
| **S1 baselines** | frozen encoders (cunei400m, AKK_300m, Llama-2-7B, Qwen3-8B) → period probes + UMAP/silhouette + source probe on all labelled sets | ~1 day | `S1_REPRESENTATION_BASELINES.md`; gate: period probe ≫ chance on ≥ 2 corpora |
| **S2 SSL sweep** | Barlow / BYOL / JEPA / InfoNCE × {adapter on frozen encoder; from-scratch 0.25–16 M} × 4 epochs-equivalent repeats | ~2–3 days | loss curves, period probe vs size/objective; gate: at least one objective beats frozen baseline on period probe |
| **S3 fine-tune sweep** | head sizes × data fractions × seeds on the 1,193 dated texts, from each S2 checkpoint | ~1 day | ρ-vs-labels curves: does pretraining cut the labels needed? |
| **S4 analysis** | scaling curves (size, data, repeats), source-bias report, held-out-source test | 0 | `SCALING_RESULT.md` |
| **S5 paper** | figures from `results.parquet`; PAPER_NOTES update | 0 | draft sections |

Everything runs through the existing runner and result store; every job pushes
its log; read-outs use the SLA §7 protocol.

## 7. Risks

* **Coarse, source-correlated period labels** → mitigated by within-source probes
  and held-out-source evaluation.
* **Tiny fragments dominate eBL** → min-length filter; report by length bucket.
* **Transliteration conventions differ** → sign-level tokens; a convention probe.
* **Data-constrained regime** → more parameters will not help from scratch; the
  win must come from pretrained encoders + better objectives, and we say so.
* **Duplicates across sources** → content-hash dedupe before splitting.

## 8. First concrete step (S0, ≈ 1 day)

`chrono/scripts/make_ssl_corpus.py`: read the unified corpus + letters + SEAL +
ORCC, rebuild texts from sign tokens per fragment, hash-dedupe, filter ≥ 8 words,
attach period/genre/provenance where known, assign tablet-grouped splits with
source balance, write `corpus_all.parquet` and a census. Then S1 on the runner.
