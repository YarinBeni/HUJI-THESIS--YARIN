# Round 3 — Results by Test (balanced vs imbalanced, every model)

Each section: **what the test is** (2 lines) → **data & split** → **full results table**.
Numbers are read from the committed result files via
`v_1/src/linear_probing/build_balance_scoreboard.py` (saves `balanced_mc_scoreboard.{json,csv}`),
plus the geodesic/LORO JSONs. All "best layer/config" entries are the best-scoring config for
that model (slightly optimistic — best-of-many-configs — but applied equally to every model).

**Models:** `thalesian_cunei400m` / `thalesian_akk300m` = Akkadian-finetuned encoders ·
`mlm` = small Akkadian MLM (Aeneas) · `tfidf` = char-n-gram baseline (spelling only, no
semantics) · `qwen` = Qwen2.5-7B (no Akkadian) · `qwen3_1b7/8b/32b` = Qwen3 scale sweep (no
Akkadian) · `random` = random-initialized control.

**What "balanced" means:** the corpus is ~95% Neo-Assyrian (≈650 BCE). *Imbalanced / full-set* =
all 1,193 year-labeled fragments as-is. *Balanced MC* = 200 random draws, each 168 fragments =
8 rulers × 21 each, scores reported mean ± std over the 200 draws — this neutralizes the
"just guess ≈650 BCE" shortcut.

---

## TEST 1 — Year regression, PLS
**What it is:** PLS (Partial Least Squares) finds a few directions in the model's activation
vectors that best predict the year, then regresses year on them. Supervised; "best layer" is
the model layer whose activations predict year best.
**Data & split:** activations of each fragment (mean-pooled). **5-fold GroupKFold grouped by
ruler** — every fold trains on some rulers and tests on *entirely held-out rulers* (no ruler in
both train and test). Metric = Spearman (rank-correlation of predicted vs true year, −1..1).

| Model | Full-set (imbalanced) Sp | best layer/cfg | Balanced Sp ± std | Δ (bal−full) |
|---|---|---|---|---|
| mlm | −0.115 † | L2 tier0/mean k2 | **0.424 ± 0.059** | +0.539 † |
| thalesian_cunei400m | 0.467 | L12 tier0/mean k2 | **0.411 ± 0.064** | −0.056 |
| tfidf | — ‡ | — | 0.407 ± 0.064 | — |
| qwen3_32b | 0.510 | L26 tier0/mean k3 | 0.399 ± 0.063 | **−0.112** |
| qwen | 0.121 † | L5 tier0/mean k5 | 0.398 ± 0.079 | +0.277 † |
| qwen3_1b7 | 0.484 | L6 tier0/mean k3 | 0.371 ± 0.081 | **−0.113** |
| qwen3_8b | 0.482 | L26 tier0/mean k3 | 0.365 ± 0.068 | **−0.117** |
| thalesian_akk300m | 0.435 | L7 tier0/mean k3 | 0.344 ± 0.062 | −0.090 |
| random | 0.184 † | L12 tier0/mean k2 | — ‡ | — |

† Degenerate full-set measurement: `mlm`/`qwen`/`random` PLS year on the imbalanced set is
near-zero or negative (GroupKFold folds collapse — a held-out ruler often spans one date). The
huge "+Δ" is *not* "balancing helped"; it's that the full-set number was broken for these
models. Read their balanced number as the real one.
‡ Coverage gap: `tfidf` full-set PLS-year and `random` balanced PLS-year were never computed.

**Bottom line:** among models with a *valid* full-set number (the strong ones), balancing
**lowers** every score, and lowers the **biggest Qwen3 models most** (−0.11 to −0.12) while
domain-finetuned Thalesian barely moves (−0.056). The imbalanced ranking (qwen3_32b 0.510 >
Thalesian 0.467) collapses into a **tie** (Thalesian 0.411 ≈ qwen3_32b 0.399, CIs overlap).
TF-IDF (0.407) and the small MLM (0.424) sit right in that same tie. **Scale's apparent
advantage was an imbalance artifact.**

---

## TEST 2 — Year regression, Ridge
**What it is:** plain Ridge (L2-penalized linear regression) predicting year from the activation
vector directly — a simpler, single-direction readout than PLS. Same supervised setup.
**Data & split:** same as Test 1 — mean-pooled activations, **5-fold GroupKFold by ruler**,
metric = Spearman.

| Model | Full-set Sp | best layer/cfg | Balanced Sp ± std | Δ (bal−full) |
|---|---|---|---|---|
| mlm | — ‡ | — | **0.408 ± 0.061** | — |
| tfidf | — ‡ | — | 0.355 ± 0.069 | — |
| qwen3_1b7 | 0.444 | L2 tier0/mean | 0.352 ± 0.068 | −0.091 |
| qwen3_8b | 0.439 | L2 tier0/mean | 0.332 ± 0.072 | −0.107 |
| qwen | — ‡ | — | 0.327 ± 0.069 | — |
| qwen3_32b | 0.429 | L62 tier0/mean | **0.326 ± 0.069** (last) | −0.103 |

‡ Full-set Ridge was only run for the qwen3_* models; mlm/tfidf/qwen have balanced-only numbers.

**Bottom line:** on the simpler Ridge readout the ordering **inverts** — a small Akkadian MLM
(0.408) and even char-n-gram TF-IDF (0.355) **beat every Qwen3**, and the 32B model is **dead
last** (0.326). Bigger model ⇒ *worse* single-direction dating; scale spreads the signal across
many correlated dimensions that one Ridge direction can't capture.

---

## TEST 3 — Ruler classification
**What it is:** predict *which ruler* a fragment belongs to (multi-class). This is the "can you
identify the king" task — names live here.
**Data & split:** **5-fold StratifiedKFold** (same rulers in train and test, balanced per fold).
Metric = Macro-F1 (per-ruler F1 averaged equally). ⚠ Imbalanced and balanced are **not**
apples-to-apples: imbalanced = 11–41 rulers (chance tiny), balanced = 8 rulers (chance 0.125),
so balanced Macro-F1 is mechanically higher — use it to *rank methods*, not to claim "balancing
helped."

| Model | Imbalanced R1 Macro-F1 | Balanced MC Macro-F1 ± std | best layer/cfg |
|---|---|---|---|
| tfidf (tier0) | 0.326 | **0.650 ± 0.037** | L00 tier0 |
| mlm | 0.220 | 0.460 ± 0.044 | L15 tier0/mean |
| thalesian_cunei400m | 0.210 | 0.448 ± 0.043 | L12 tier0/mean |
| qwen3_8b | — | 0.369 ± 0.040 | L00 tier0/mean |
| qwen (Qwen2.5-7B) | 0.117 | 0.363 ± 0.042 | L03 tier0/mean |
| qwen3_32b | — | 0.359 ± 0.039 | L06 tier0/mean |
| qwen3_1b7 | — | 0.354 ± 0.039 | L00 tier0/mean |
| thalesian_akk300m | — | 0.323 ± 0.039 | L08 tier0/mean |

**Bottom line:** **TF-IDF wins ruler-ID outright** (0.650, far above any neural model). Spelling
alone identifies the king better than a 32B LLM. Again no neural-scale benefit.

---

## TEST 4 — Geodesic / Isomap manifold (unsupervised)
**What it is:** instead of *training* a probe, we ask whether the fragments *already* lie along a
curved 1-D "timeline" in activation space. **Isomap** = a manifold method: build a k-nearest-
neighbor graph on the vectors, then "unroll" it into one coordinate. We never show it the years;
we just check whether that unrolled coordinate happens to order the texts by date.
**Data & split:** **no labels used to fit** (unsupervised); evaluated on **all** fragments.
Metric = **pacc** (pairwise-order accuracy, ±100yr margin): of all fragment pairs >100 years
apart, the fraction the 1-D coordinate orders correctly. 0.5 = coin-flip, 1.0 = perfect.

| Model | Best pacc | geodesic Spearman | layer/cfg |
|---|---|---|---|
| **qwen (Qwen2.5-7B)** | **0.731** | 0.332 | L1 maximal/mean |
| qwen3_1b7 | 0.723 | 0.250 | L1 tier0/mean |
| qwen3_32b | 0.716 | 0.310 | L1 maximal/mean |
| qwen3_8b | 0.716 | 0.316 | L1 maximal/mean |
| thalesian_cunei400m | 0.681 | 0.243 | L7 maximal/mean |
| thalesian_akk300m | 0.662 | 0.185 | L0 tier0/mean |

**Bottom line:** the best *unsupervised* timeline belongs to **qwen (Qwen2.5-7B), L1, pacc
0.731** — the model with the *worst supervised* dating. The temporal manifold lives in the early
(near-token-embedding) layers and tracks lexical/orthographic drift; deeper layers dilute it.
This is the one genuinely interesting neural result of the round — and it's geometric, not
predictive.

---

## TEST 5 — LORO (Leave-One-Ruler-Out honesty pass)
**What it is:** is the manifold a real *timeline*, or just "each ruler is its own blob that
happens to sit near its date"? We refit the Isomap manifold with **one ruler's fragments
removed**, then drop those held-out fragments onto it and re-measure pacc. Small drop = genuine
temporal axis; big drop = ruler-cluster artifact.
**Data & split:** held-out = one ruler at a time (11 rulers); manifold fit on the other 10.
Metric = pacc drop (full minus mean-over-held-out-rulers).

| Config | pacc (full) | pacc (LORO mean) | drop | verdict |
|---|---|---|---|---|
| qwen maximal/mean L1 | 0.731 | 0.723 | **+0.008** | STRONG |
| thalesian_cunei400m tier0/mean L6 | 0.645 | 0.617 | +0.029 | STRONG |
| thalesian_cunei400m maximal/mean L7 | 0.681 | 0.626 | +0.055 | STRONG |

**Bottom line:** all three drops are tiny (<0.06) → the manifold is a **genuine temporal axis**,
not ruler-cluster geometry. Held-out rulers land at approximately the right date.

---

## TEST 6 — Phase D visualization (centroid + spline)
**What it is:** a sanity/figure check — bin fragments into 100-year windows, take each window's
centroid in 3-D PCA space, fit a smooth curve (spline) through the centroids, and measure
whether distance-along-the-curve tracks century order.
**Data & split:** all fragments, 7 populated century-bins. Metric = arc-length Spearman.

| Config | arc-length Spearman | # bins |
|---|---|---|
| qwen maximal/mean L1 | **1.000** | 7 |
| thalesian_cunei400m maximal/mean L7 | **1.000** | 7 |
| thalesian_cunei400m tier0/mean L6 | **1.000** | 7 |

**Bottom line:** century centroids thread in perfect chronological order for all configs (caveat:
only 7 centroids, 728/992 fragments sit in the 650-BCE bin). Plots embedded in
`EXPERIMENTS_SUMMARY.md`.

---

## TEST 7 — TF-IDF name-masking control
**What it is:** does TF-IDF date texts by reading the *king's name*, or by period spelling? We
mask all personal names (`m-`/`f-` determinative tokens + theophoric `d-god-predicate`
sentence-names like Nabû-kudurri-uṣur=Nebuchadnezzar → `[PN]`; bare gods kept) and re-date.
**Data & split:** balanced MC (same 200 draws). Year via Ridge GroupKFold-by-ruler → Spearman;
ruler via logistic StratifiedKFold → Macro-F1.

| Cleaning | Condition | Year Spearman | Ruler Macro-F1 |
|---|---|---|---|
| tier0 | unmasked | 0.355 ± 0.069 | 0.650 ± 0.037 |
| tier0 | **masked** | **0.400 ± 0.062** | **0.527 ± 0.041** |
| maximal | unmasked | 0.266 ± 0.078 | 0.498 ± 0.040 |
| maximal | masked | 0.268 ± 0.086 | 0.463 ± 0.039 |

**Bottom line:** masking *all* names drops ruler-ID by −0.122 but leaves **dating unchanged**
(0.355→0.400, within CI). Dating is **orthographic drift, not name lookup** — and post-mask
feature re-ranking confirms zero king names remain in the top dating features.

---

## OVERALL — did balancing help?

**No model is *helped* by balancing — that's the wrong frame.** Balancing is a *fairness
correction*: it removes the Neo-Assyrian shortcut so the numbers reflect real dating skill, not
"guess 650 BCE." It does three things:

1. **Lowers every (valid) score**, because the easy mass is gone.
2. **Lowers the biggest models most** (Qwen3 −0.11; Thalesian −0.06) → the full-set
   "scale wins" (qwen3_32b 0.510 > Thalesian 0.467) **collapses into a tie** (0.40 ≈ 0.41).
3. **Inverts the Ridge ranking**: small MLM (0.408) and TF-IDF (0.355) beat every Qwen3; the 32B
   is last (0.326).

**Thesis takeaway:** once imbalance is removed, **scale does not buy dating** — a char-n-gram
baseline ties a 32B LLM and a domain-finetuned encoder. Dating Akkadian is *shallow orthographic
drift* (Tests 2, 3, 7). The only genuine neural win is *geometric* — the unsupervised temporal
manifold (Tests 4, 5), strongest in a model with **no** Akkadian training.
