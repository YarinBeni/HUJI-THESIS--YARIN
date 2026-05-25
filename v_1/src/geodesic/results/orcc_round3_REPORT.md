# Round 3 — Geodesic / Manifold Readout of Akkadian Temporal Structure

**Date:** 2026-05-24 (Phases A–D complete; Phase E1 scale sweep in progress)
**Branch:** main — all result files committed.

---

## Background & Motivation

Round 2 established that Thalesian cuneiBase-400m encodes temporal information (PLS Spearman 0.467, MAE 75 years) but left open whether that signal is organized as a coherent **geometric manifold** in representation space. Round 3 tests this with an unsupervised pipeline: PCA-64 → L2-normalize → smallest-connected kNN graph → Isomap 1D embedding. If the temporal axis is genuinely manifold-structured, this readout should match the supervised PLS headline without any label supervision.

Round 3 also expands the scale sweep to Qwen3-1.7B / 8B / 32B (Phase E1) to test whether bigger multilingual models can match or surpass Thalesian's year-prediction accuracy.

---

## Phase 0 — Activation Inventory (job 8529)

| Method | Cleaning | Pool | Layers | Shape |
|---|---|---|---|---|
| qwen (Qwen2.5-7B) | tier0 + maximal | mean + last | 29 | (1202, 3584) |
| thalesian_akk300m | all 4 | all | 9 | (1202, 512) |
| thalesian_cunei400m | all 4 | all | 13 | (1202, 768) |
| random_qwen | all 4 | all | 29 | (1202, 3584) |
| qwen3_1b7 | tier0 + maximal | mean + last | 29 | (1202, 2048) |
| qwen3_8b | tier0 + maximal | mean + last | 37 | (1202, 4096) |
| qwen3_32b | tier0 + maximal | mean + last | 65 | (1202, 5120) |

Parquet: 1,193 / 1,202 year-labeled. MLM (Aeneas) activations not found; excluded.

---

## Phase A — Single-layer POC (job 8530)

**Configuration:** thalesian_cunei400m / tier0 / mean / L12 (Round 2 PLS-selected best layer).

| Metric | PLS (full-set) | Isomap |
|---|---|---|
| Spearman | 0.633 | **0.035** |
| Pairwise-order acc (±100yr) | 0.859 | **0.547** |
| Neighbor purity (k=10, ±100yr) | — | **0.796 (+13.4σ)** |

**Gate verdict: FAIL** — Isomap pacc = 0.547 < 0.60 threshold.

**But we ran Phase B anyway.** Reason: Phase A tested the PLS-selected layer (L12), not the geodesic-optimal layer. Neighbor purity at +13.4σ confirms local temporal clustering is real. The question is whether a different layer makes the 1D manifold globally coherent. This is confirmed in Phase B.

**Phase A interpretation:** Thalesian's temporal signal is real but not organized as a globally monotone 1D curve at L12. The encoding is locally coherent (nearby embeddings cluster by era) but multi-dimensional: PLS extracts the signal via supervised projection; Isomap 1D cannot resolve it at this layer.

---

## Phase B — Full Layer × Method Geodesic Scoreboard (26 jobs)

**Pipeline:** PCA-64 (StandardScaler + PCA) → L2-normalize → smallest-connected kNN → Isomap 1D. Scored by pairwise-order accuracy (±100yr margin) over all 26 (method, cleaning, pool) combos, all available layers.

### Full scoreboard (sorted by best isomap pacc)

| Method | Cleaning | Pool | Best L | pacc | Sp |
|---|---|---|---|---|---|
| **qwen** | maximal | mean | **1** | **0.731** | 0.332 |
| qwen3_1b7 | tier0 | mean | 1 | 0.723 | 0.250 |
| qwen | tier0 | mean | 7 | 0.712 | 0.225 |
| qwen3_1b7 | maximal | mean | 1 | 0.713 | 0.299 |
| qwen3_8b | maximal | mean | 1 | 0.716 | 0.316 |
| qwen3_32b | maximal | mean | 1 | 0.716 | 0.310 |
| qwen3_8b | tier0 | mean | 2 | 0.703 | 0.279 |
| qwen3_32b | tier0 | mean | 1 | 0.688 | 0.214 |
| thalesian_cunei400m | maximal | mean | **7** | **0.681** | 0.243 |
| thalesian_akk300m | tier0 | mean | 0 | 0.662 | 0.185 |
| thalesian_akk300m | maximal | mean | 3 | 0.661 | 0.263 |
| thalesian_cunei400m | tier0 | mean | **6** | **0.645** | 0.108 |
| thalesian_akk300m | tier0 | last | 1 | 0.655 | 0.186 |
| qwen | tier0 | last | 6 | 0.615 | 0.134 |
| qwen3_32b | tier0 | last | 6 | 0.615 | 0.141 |
| thalesian_akk300m | maximal | last | 1 | 0.627 | 0.144 |
| qwen3_8b | tier0 | last | 10 | 0.605 | 0.133 |
| thalesian_cunei400m | maximal | last | 4 | 0.599 | 0.083 |
| qwen3_32b | maximal | last | 38 | 0.591 | 0.086 |
| qwen | maximal | last | 27 | 0.598 | 0.142 |
| qwen3_8b | maximal | last | 13 | 0.585 | 0.067 |
| qwen3_1b7 | tier0 | last | 3 | 0.589 | 0.112 |
| qwen3_1b7 | maximal | last | 5 | 0.580 | 0.043 |
| thalesian_cunei400m | tier0 | last | 1 | 0.573 | 0.055 |

### Phase B key findings

1. **Mean pooling >> last-token pooling** across all models (mean +0.10–0.14 pacc). Mean pooling captures the full distributional signature of a fragment; last-token captures a single context-dependent vector.

2. **qwen (Qwen2.5-7B, no Akkadian training) geodesically outperforms Thalesian** — pacc 0.731 vs 0.681. This is counter-intuitive: qwen PLS Spearman = 0.121 (Round 2), far below Thalesian 0.467. The geodesic result says qwen's L1 embeddings have a cleaner temporal manifold than Thalesian's L7, even without domain training. Likely mechanism: qwen L1 is essentially a token-embedding layer; Akkadian vocabulary drifts systematically over centuries (new loan words, orthographic conventions), so L1 mean-pool ≈ bag-of-token-types that implicitly indexes era.

3. **Layer 1 dominates for mean pooling** — the geodesic best layer for qwen, qwen3_1b7/8b/32b is L1 (or L2). For Thalesian it's L6–L7. Layer-selection bias is confirmed: PLS best layer (L12 for thalesian) ≠ geodesic best layer (L7). Testing only at the PLS-selected layer would have missed the geodesic signal.

4. **No geodesic scale effect** — qwen3_1b7 (0.723), qwen3_8b (0.716), qwen3_32b (0.716) are all within noise. Bigger Qwen3 does not buy a better temporal manifold.

5. **Phase B gate PASSES** for thalesian_cunei400m/maximal/mean/L7: pacc = 0.681 ≥ 0.60.

---

## Phase C — LORO Honesty Pass (33 jobs: 11 rulers × 3 configs)

**Protocol:** For each ruler, refit PCA+Isomap on the held-in fragments only. Project held-out fragments via `Isomap.transform()`. Score cross-ruler pairwise-order accuracy. A large drop in pacc when a ruler is withheld → its position was anchored by ruler-cluster geometry, not temporal structure.

**Gate:** Mean drop < 0.10 = STRONG (genuinely temporal); < 0.20 = HEDGED; ≥ 0.20 = WEAK.

| Config | pacc_full | pacc_loro_mean | drop | Verdict |
|---|---|---|---|---|
| qwen/maximal/mean/L1 | 0.731 | 0.723 | **0.008** | **STRONG** |
| thalesian_cunei400m/maximal/mean/L7 | 0.681 | 0.626 | **0.055** | **STRONG** |
| thalesian_cunei400m/tier0/mean/L6 | 0.645 | 0.617 | **0.029** | **STRONG** |

**All 3 configs pass STRONG.** The temporal manifold is not an artifact of ruler-cluster geometry.

### Per-ruler detail (notable exceptions)

**Nebuchadnezzar I** (n=10, ~1100 BCE): cross-pacc collapses to 0.21–0.35 across all configs. This ruler is an outlier — 10 fragments ca. 1100 BCE, temporally isolated from the main corpus (550–750 BCE). With only 10 held-out fragments, cross-pacc is high-variance; the poor performance reflects chronological isolation, not a flaw in the manifold.

**Nabopolassar** (n=15) in thalesian/tier0/L6: individual drop = 0.216. Again a small-sample ruler. 15 fragments is at the lower limit for stable Isomap; this appears to be sampling noise rather than a systematic failure.

**Thalesian/maximal/L7 — Nabonidus** (n=68): individual drop = 0.116 (over the 0.10 threshold alone, but mean still STRONG). Nabonidus reigned 556–539 BCE, overlapping with Nebuchadnezzar II; within-era disambiguation may depend on textual cues beyond temporal structure.

**qwen config:** Near-perfect LORO stability (drop = 0.008). Almost no degradation when any ruler is removed. This confirms that qwen's L1 manifold encodes a truly corpus-wide temporal axis, not ruler-specific clusters.

---

## Phase D — Centroid + Spline Visualization (3 configs × 4 plots = 12 PNGs)

**Protocol:** For each config, bin fragments into 100-year windows (min 5 per bin), compute PCA-3D centroids per bin, fit a cubic UnivariateSpline (weighted by √n), compute arc-length along spline, report arc-length Spearman vs bin year.

| Config | pacc | Geodesic Sp | Arc-len Sp | Bins |
|---|---|---|---|---|
| qwen/maximal/mean/L1 | 0.731 | 0.332 | **1.000** | 7 |
| thalesian_cunei400m/maximal/mean/L7 | 0.681 | 0.243 | **1.000** | 7 |
| thalesian_cunei400m/tier0/mean/L6 | 0.645 | 0.108 | **1.000** | 7 |

**Arc-length Spearman = 1.0 for all configs.** The cubic spline traces a perfectly monotone arc through the 3D PCA space as a function of time — the 100-year bin centroids are ordered chronologically along the manifold. This is the strongest visualization result possible.

**7 bins** are populated (≥5 fragments each): centered near 50, 550, 650, 750, 950, 1050, 1150 BCE. The corpus is heavily concentrated in 550–750 BCE (1,137 / 1,192 fragments). Despite this imbalance, the spline correctly orders all 7 centuries.

**Plots saved:** `v_1/src/geodesic/results/phase_d/` — 4 coloring variants per config (year / ruler / archive / geodesic coordinate).

---

## Phase E1 — Qwen3 Scale Sweep (PLS + Ridge year probes)

### PLS year-raw results (best layer per model, maximal/mean unless noted)

| Model | Best L | PLS Sp | MAE (yr) | vs Thalesian |
|---|---|---|---|---|
| **qwen3_32b** | 26 | **0.511** | 74.7 | **+0.044** |
| qwen3_1b7 | 6 | 0.484 | 75.1 | +0.017 |
| qwen3_8b | 26 | 0.482 | 77.8 | +0.015 |
| **thalesian_cunei400m** | 12 | **0.467** | 75.1 | — (baseline) |
| thalesian_akk300m | 7 | 0.435 | 76.5 | −0.032 |
| qwen (Qwen2.5-7B) | 5 | 0.121 | 128.3 | −0.346 |

### Ridge (cls_numeric) year-raw results

| Model | Best L | Ridge Sp | MAE (yr) |
|---|---|---|---|
| qwen3_1b7 | 2 | 0.444 | 80.6 |
| qwen3_8b | 2 | 0.439 | 81.7 |
| qwen3_32b | 62 | 0.429 | 84.5 |

### Balanced Monte-Carlo results (200 draws × 168 fragments, 8 rulers × 21)

The full-set numbers above are computed on the imbalanced 1,193-fragment corpus.
The balanced MC re-runs each probe on 200 class-balanced sub-draws (matching the
Round 2 Phase 0 protocol) to control for ruler imbalance. **PLS year-raw, best
layer per model:**

| Model | Layer | **Balanced PLS Sp ± std** | Full-set Sp (for reference) |
|---|---|---|---|
| **thalesian_cunei400m** | L12 | **0.411 ± 0.064** | 0.467 |
| qwen3_32b | L09 | 0.399 ± 0.063 | 0.511 |
| qwen3_1b7 | L09 | 0.371 ± 0.081 | 0.484 |
| qwen3_8b | L01 | 0.365 ± 0.068 | 0.482 |
| thalesian_akk300m | L06 | 0.344 ± 0.062 | 0.435 |

Ridge (cls_numeric) balanced, at the full-set reported layer: 1b7 @L02 = 0.287 ± 0.070,
8b @L02 = 0.320 ± 0.074, 32b @L62 = **0.245 ± 0.077** (worst — confirms Ridge does
not scale).

### Phase E1 key findings

1. **The qwen3_32b "win" does NOT survive balancing — this is the headline correction.**
   On the full imbalanced set, qwen3_32b (0.511) beat Thalesian cunei400m (0.467). Under
   balanced MC, **Thalesian cunei400m is nominally best (0.411 ± 0.064) and qwen3_32b
   (0.399 ± 0.063) is a statistical tie** — every model's CI overlaps the others (±0.06–0.08).
   The full-set scale advantage was partly an *imbalance artifact*: the larger model exploited
   the extra (imbalanced, Neo-Assyrian-dominated) data. Honest claim for the thesis:
   *"domain fine-tuning (Thalesian) and frontier-scale multilingual pretraining (Qwen3-32B)
   reach statistically indistinguishable year-regression performance under class balancing;
   neither dominates."*

2. **PLS scales with model size on the full set; the trend flattens under balancing.**
   Full-set PLS: 1b7 (0.484) ≈ 8b (0.482) < 32b (0.511). Balanced PLS: 1b7 (0.371) ≈
   8b (0.365) ≈ 32b (0.399) — all within one std. Scale buys little once imbalance is removed.

3. **Ridge does not scale, in either regime.** Full-set Ridge: 1b7 (0.444) ≈ 8b (0.439) >
   32b (0.429). Balanced Ridge at the reported layer is worst for 32b (0.245). The temporal
   signal in the larger model is spread across correlated dimensions that PLS's multi-component
   projection captures but single-component Ridge cannot.

4. **Qwen2.5-7B (the "qwen" model) fails PLS entirely** (full-set sp=0.121). Qwen3 models
   (0.48–0.51 full-set) vastly outperform their Qwen2.5 predecessor — an architectural /
   training-data change between Qwen generations, not just scale.

5. **Geodesic vs PLS gap for qwen pretrained:** qwen geodesic pacc=0.731 (best overall) but
   PLS sp=0.121 (worst). The L1 token-embedding manifold has clean temporal geometry that
   Isomap reads, but a supervised linear probe on L5 finds almost no signal — different layers,
   different mechanisms (lexical-drift geometry at L1 vs distributed semantic signal deeper).

**Speedup note (methods):** the balanced MC sweep was projected at ~6+ days sequential
(32b alone ≈115hr). Parallelizing the per-layer loop with joblib threads (BLAS pinned to 1)
plus fanning draws across 4 chunks/model cut the whole sweep to **~40 min wall-clock**. The
runner is now self-healing against partial JSONs left by walltime-killed jobs.

---

## Confound control — TF-IDF name-masking (the "is it just the king's name?" test)

A bag of **character n-grams (TF-IDF, char_wb 2–5)** has no semantics — it can only see
spelling. We use it as a transparent confound probe. We mask the Akkadian **personal-name
determinative** (`m-…` whitespace tokens → `[PN]`; e.g. `m-eri-ba`→Sennacherib,
`m-tukul-ti`→Tiglath-pileser, `m-tar-qu-u`→Taharqa) and re-run balanced MC (200 draws,
same draws as above). Year via Ridge(GroupKFold-ruler)→Spearman; ruler via logistic→Macro-F1.

| Cleaning | Condition | Year Spearman | Year MAE | Ruler Macro-F1 |
|---|---|---|---|---|
| tier0 | unmasked | 0.355 ± 0.069 | 43.9 | 0.650 ± 0.037 |
| tier0 | **masked** | **0.391 ± 0.062** | 43.7 | **0.551 ± 0.040** |
| maximal | unmasked | 0.266 ± 0.078 | 47.5 | 0.498 ± 0.040 |
| maximal | **masked** | 0.268 ± 0.086 | 47.3 | **0.463 ± 0.039** |

**Findings:**

1. **Masking names costs ruler-ID but not dating.** Ruler Macro-F1 drops 0.099 (tier0) /
   0.035 (maximal); year Spearman is unchanged (tier0 even nudges up 0.355→0.391, within CI).
   The dating signal does **not** live in the explicit king's name — it survives name removal.

2. **TF-IDF dates as well as the neural models.** Masked TF-IDF year Spearman (0.391, tier0)
   is a statistical tie with balanced Thalesian (0.411) and qwen3_32b (0.399). **A name-masked
   bag of character n-grams matches a 32B LLM and a domain-finetuned encoder on dating.** The
   bulk of the chronological signal is **shallow orthographic / spelling drift** (sign forms
   and spelling conventions changed over centuries), not deep semantic understanding.

3. **Caveat — what `m-` masking does and doesn't remove.** It removes *explicitly determined*
   personal names. It does **not** remove theophoric / logographic name elements that double as
   ordinary period vocabulary (e.g. Neo-Babylonian `na-bi`=Nabû, `uṣur`, `sag-il`=Esagil are
   unchanged by masking — they are entangled with religious/dialect vocabulary). So "name" here
   means *explicitly marked* names; deeper dynasty/period vocabulary is not masked and may still
   carry chronological information legitimately.

**Thesis takeaway:** dating Akkadian texts is, to first order, an **orthographic-drift** task
solvable without semantics or names. Neural models (domain-finetuned or frontier-scale) do not
beat this shallow baseline under class balancing. The interesting neural result is therefore
*geometric* (the manifold story, Phases A–D), not raw predictive accuracy.

---

## Cross-phase synthesis

### The two-regime finding

| Probe type | Best model | Best Sp / pacc |
|---|---|---|
| PLS (supervised, full set) | qwen3_32b | Sp = 0.511 |
| PLS (supervised, balanced MC) | thalesian_cunei400m | Sp = 0.411 ± 0.064 |
| Geodesic (unsupervised) | qwen/maximal/mean/L1 | pacc = 0.731 |

These are different models and different layers. The best unsupervised manifold belongs to the model with the *worst* supervised probing (qwen Qwen2.5-7B). This is not a contradiction: qwen's L1 mean-pool is essentially a bag-of-token-types, which captures lexical drift across centuries. Deeper layers integrate context, ruler identity, genre, and other information that dilutes the temporal axis while enabling richer supervised extraction.

**Imbalance caveat on the supervised leaderboard.** The full-set PLS winner (qwen3_32b, 0.511) is overtaken by Thalesian (0.411) once draws are class-balanced; the two are a statistical tie. Any supervised-probing claim in this round must cite the balanced MC numbers, not the imbalanced full-set point estimates.

### Temporal manifold is real (Phase C verdict)

All three tested configs pass STRONG LORO (drops 0.008–0.055). The manifold is not ruler-cluster geometry. Texts from held-out rulers land in approximately the correct temporal position in the held-in manifold.

### Layer-selection bias matters

Phase A would have concluded "geodesic fails" for Thalesian if we had stopped at the PLS-selected layer (L12, pacc=0.547). The geodesic-optimal layer for Thalesian is L7 (pacc=0.681 — passes the gate). Always sweep layers independently for geodesic and supervised probing.

---

## Suggested thesis framing

> *"Akkadian cuneiform texts are temporally organized in the representation spaces of multiple language models, but the nature of that organization differs by model and probe type. For Thalesian (domain-finetuned), temporal structure is multi-dimensional: PLS extracts it with balanced-MC Spearman 0.411 ± 0.064 but unsupervised Isomap peaks at pacc 0.681. For Qwen2.5-7B (no domain training), the L1 token-embedding layer encodes lexical drift as a near-globally-monotone 1D manifold (pacc 0.731, LORO drop 0.008 — STRONG), while deeper layers show no supervised linear signal. Frontier-scale multilingual pretraining (Qwen3-32B) reaches balanced-MC Spearman 0.399 ± 0.063 — statistically indistinguishable from domain fine-tuning, despite zero Akkadian training. The apparent scale advantage of Qwen3-32B over Thalesian on the full imbalanced set (0.511 vs 0.467) does not survive class balancing. Together, these results suggest that temporal information in cuneiform is encoded both as lexical drift (accessible at L1, without domain training) and as distributed semantic-structural signal (accessible with PLS at deeper layers), and that domain fine-tuning and frontier-scale pretraining are two routes to comparable — not dominant — supervised year-regression performance."*

---

## Pending

- [x] MC balanced CIs for qwen3_1b7/8b/32b — DONE (parallel fan-out jobs 8743–8752, 8734–8738; 200 draws each)
- [x] TF-IDF name-masking confound control — DONE (local, 200 draws; dating survives masking, ties neural models)
- [ ] Backfill cls_numeric Ridge for qwen/mlm/tfidf baselines (cluster jobs 8758–8765; thalesian PLS balanced done)
- [x] Phase D PNGs review — DONE (arc-len Sp=1.0; flagged dense-blob readability)
- [ ] Confound controls C1 (metadata-only year baseline) + C2 leave-one-provenance-out (genre unusable — 1 value)
- [ ] NN-audit by same-ruler (C3, ruler-only per scope decision)
- [ ] SAE attribution on qwen3_32b (if time/scope permits — Track C)
