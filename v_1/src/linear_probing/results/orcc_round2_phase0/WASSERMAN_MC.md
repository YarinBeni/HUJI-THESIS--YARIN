# Wasserman & Ni MC Protocol — Extraction for ORCC Round 2 Phase 0

**Source paper:** Wasserman, N. & Ni, C., "Chronological Attribution and Genre Cohesion Through
Computational Lexicometry: A UMAP-Monte Carlo Analysis for Akkadian Corpora" (REVISED ms., 2026)

**Extracted by:** W1.A agent, 2026-05-19, for ORCC Round 2 Phase 0 gate criteria.

---

## 1. Sampling Protocol

### What MCS does (§ 2.3)

Wasserman & Ni's Monte Carlo Simulation (MCS) is a **word-level bootstrap / augmentation
procedure**, NOT subsampling for classification evaluation. The algorithm is:

```
For each corpus Ci (i = 1..N):
    For each virtual text j = 1..n_txt:
        For each word slot k = 1..n_word:
            Sample word w from Ci with probability proportional to
            its original frequency in Ci.
        Concatenate n_word words → virtual text VTj
    Virtual corpus VCi = {VT1, …, VT_n_txt}
```

**Key parameters used in the paper:**

| Section / Figure | n\_txt | n\_word | Corpus |
|---|---|---|---|
| Fig. 2.5 (RINA test) | 800 | varies (Fig shows range) | Neo-Assyrian RI |
| Fig. 2.7 (n\_txt sensitivity) | 400–700 | 100 | RINA |
| Fig. 2.8 (hyperparameter stability) | 600 | 100 | RINA |
| Fig. 3.1b (Large Akkadian) | 3,000 | 60 | All large corpora |
| Fig. 3.2b (RINA final) | 600 | 100 | RINA |
| Fig. 3.3b (Neo-Babylonian RI) | 300 | 80 | NBRI |
| Fig. 3.6b (Erra & Išum) | 400 | 100 | SB+OB+LB lit |

**Constraint noted explicitly (§ 2.3):** `n_word × n_txt` must be larger than the size of the
largest actual corpus being equalized.

**Purpose of the procedure:** equalize imbalanced corpus sizes so UMAP does not treat small
sub-corpora as noise. It is word-sampling WITH replacement (weighted by original frequency), and
preserves the original word distribution.

**IDF note (§ 2.3):** When computing TF-IDF on MCS texts, the IDF is calculated from the
**original (non-virtual) corpus**, not from the synthetic texts. This is critical — MCS texts
only replace the TF component; IDF stays grounded in real data.

### This is augmentation, not undersampling

The paper does NOT use equal-size-per-class undersampling in the traditional ML sense.
MCS generates *synthetic equal-size virtual corpora* to feed into UMAP+k-NN, rather than
drawing balanced subsets from real data for classifier training/evaluation.

---

## 2. What Is Aggregated Across Draws / Reported Statistics

The paper does **not** report mean/median F1 or accuracy aggregated across MC draws in the
usual sense. MCS is used to produce a single augmented dataset (a fixed set of n_txt virtual
texts per corpus), which is then passed to UMAP+k-NN in one go.

**Classifier accuracy is reported as:** `mean ± standard deviation` over **k-fold cross-
validation** (not over MC iterations). Examples:

- Large Akkadian corpora (§ 3.1): k-NN accuracy = **0.844 ± 0.064** (k=40; 10-fold CV)
- RINA without MCS (§ 3.2): k-NN accuracy = **0.747 ± 0.073** (k=20; 5-fold CV)
- NBRI without MCS (§ 3.3): k-NN accuracy = **0.716 ± 0.069** (k=10; 5-fold CV)
- 2nd-mil. literary genres (§ 3.4): k-NN accuracy = **0.744 ± 0.012** (k=20; 5-fold CV)
- 1st-mil. literary genres (§ 3.4): k-NN accuracy = **0.880 ± 0.075** (k=10; 10-fold CV)
- Incantations (§ 3.5): k-NN accuracy = **0.938 ± 0.037** (k=40; 10-fold CV)
- Erra & Išum (§ 3.6): k-NN accuracy = **0.821 ± 0.134** (k=40; 10-fold CV)
- King of Justice (§ 3.7): k-NN accuracy = **0.875 ± 0.119** (k=10; 10-fold CV)

For the MCS validation (§ 2.3 / Fig. 2.6), they define three scores:
- Score 0: 10-fold CV k-NN accuracy on original data (baseline = 0.731 for RINA)
- Score 1 (Orig→MC): k-NN trained on original, tested on MCS data
- Score 2 (MC→Orig): k-NN trained on MCS, tested on original data
The paper states Score 1 > 0.731 baseline and Score 2 < baseline, interpreted as evidence
that MCS produces a smoother / denoised version of the original RLS space.

No confidence intervals beyond ±std are reported. No IQR or median.

---

## 3. Train / Test Split Inside Each Evaluation

The paper uses **k-fold cross-validation** exclusively for k-NN accuracy evaluation.
Two variants are used depending on corpus size:

- **10-fold CV** for larger corpora (Large Akkadian §3.1; 1st-mil. lit §3.4; incantations §3.5;
  Erra case studies §3.6–3.7)
- **5-fold CV** for smaller corpora (RINA §3.2; NBRI §3.3; 2nd-mil. lit §3.4)

No held-out test set and no leave-one-out is used. The k in k-NN and the k in k-fold are
distinct parameters — the paper is explicit about both.

For NBRI (§ 3.3), groups with fewer than 5 texts were excluded from accuracy calculation
(i.e., from being a fold label), though they are still visualized.

---

## 4. Reported TF-IDF / k-NN Accuracy Numbers — Phase 0 Targets

The paper does not frame any single number as a "TF-IDF baseline" to beat. All accuracy
numbers are k-NN classifier accuracy on TF-IDF–vectorized, UMAP-projected representations.
There is no standalone TF-IDF classification without UMAP.

**The most directly relevant number for us** is the RINA (Neo-Assyrian Royal Inscriptions)
result, which is our closest analog (same royal-inscription genre, overlapping kings):

| Setup | Metric | Value | Section |
|---|---|---|---|
| RINA, original (no MCS), n_neighbors=40, min_dist=0.2 | k-NN acc (k=20, 5-fold CV) | **0.747 ± 0.073** | §3.2, Fig. 3.2a |
| RINA, MCS augmented, n_neighbors=5, min_dist=0.3 | qualitative (no acc. reported) | — | §3.2, Fig. 3.2b |
| Large Akkadian (includes RINA) | k-NN acc (k=40, 10-fold CV) | **0.844 ± 0.064** | §3.1, Fig. 3.1a |

**UMAP dimensionality effect (Table 1):** Increasing UMAP output from 2D to 5D gives only minor
improvement; 2D is used throughout. The table reports mean ± std for dimensions 2–5 but exact
values are not reproduced in the extracted text.

### Recommended Phase 0 gate threshold

Given the RINA 5-class setup (Tiglath-pileser III, Sargon II, Sennacherib, Esarhaddon,
Ashurbanipal) achieves **0.747 ± 0.073** without MCS on TF-IDF + UMAP + k-NN:

- A bare TF-IDF linear probe (no UMAP) on our 8-class ORCC labeled set should be benchmarked
  against **~0.75 macro-accuracy** as a minimum passing threshold.
- With MCS-balanced data, we might expect closer to 0.80–0.84 (cf. Large Akkadian 0.844).
- Our task is harder (8 classes vs. 4–5, smaller per-class n) so matching 0.747 on balanced
  data is a reasonable Phase 0 gate.

---

## 5. Pitfalls and Caveats Noted in the Paper

1. **MCS smooths, does not reveal:** MCS Score 2 (MC→Orig) < baseline means the original data
   is NOT fully recoverable from the MCS representation. MCS discards fine-grained variation
   deliberately. Do not use MCS if you need to detect intra-corpus heterogeneity (§ 2.3).

2. **n_txt sensitivity is low but not zero:** Fig. 2.7 shows stability from ~400 to ~700 MCS
   texts, but below ~400 the geometry can become noisy. Use n_txt ≥ 400 as a lower bound
   (§ 2.3).

3. **n_word × n_txt > max corpus size** must hold (§ 2.3). Violating this creates virtual
   corpora that are statistically under-sampled relative to the original.

4. **IDF must come from real data:** Computing IDF from the synthetic MCS texts would distort
   the weighting of rare words, which are central to the method's discriminative power (§ 2.3).

5. **Short texts are noise, not excluded:** The paper notes that small sub-corpora (like
   Shalmaneser V, 5 texts = 502 words; Aššur-etel-ilāni, 5 texts = 502 words) are kept but
   their MCS representations are "faint" and should be interpreted cautiously (§ 3.2).

6. **Global structure vs. accuracy tradeoff:** n_neighbors = 5 maximizes k-NN accuracy but
   destroys global arrangement. n_neighbors = 40 is the balance point (§ 2.1–2.2).

7. **Metric is k-NN accuracy, not macro-F1:** The paper only reports plain accuracy (no class-
   weight adjustment). For imbalanced evaluation you may want to convert to macro-F1 separately.
   The paper does not report precision/recall/F1 anywhere.

8. **UMAP is stochastic:** The paper controls this via UMAP hyperparameter grids and MCS
   stability checks, but does not explicitly fix random seeds in the text. Results may vary
   across runs. Our implementation should set seeds.

---

## 6. Rulers / Periods Studied vs. Our ORCC Setup

### Paper's RINA corpus (§ 3.2)
| King | Texts | Total words |
|---|---|---|
| Tiglath-pileser III (745–727 BCE) | 75 | 12,222 |
| Sargon II (722–705 BCE) | 140 | 32,246 |
| Sennacherib (705–681 BCE) | 237 | 42,511 |
| Esarhaddon (681–669 BCE) | 176 | 25,378 |
| Ashurbanipal (669–631 BCE) | 266 | 60,319 |
| Aššur-etel-ilāni + Sîn-šar-iškun | 5+21 | 502+3,133 |

The RINA corpus is **Neo-Assyrian only**. Their main analysis uses **4 "large" kings** (Sargon,
Sennacherib, Esarhaddon, Ashurbanipal) and MCS is validated on these four. The last two kings
are present but treated with caution due to small size (§ 3.2).

### Our ORCC 8-ruler setup
We plan to use: Ashurbanipal, Sennacherib, Esarhaddon, Sargon II, Nebuchadnezzar II,
Tiglath-pileser III, Nabonidus, Sîn-šarru-iškun. Min class size = 21.

**Differences vs. Wasserman & Ni:**

1. **Cross-empire mixing:** We include Nebuchadnezzar II and Nabonidus (Neo-Babylonian), while
   RINA is purely Neo-Assyrian. Wasserman & Ni treat these in a completely separate section
   (§ 3.3, NBRI) and find less clear chronological structure there (0.716 ± 0.069 vs. 0.747).
   Mixing NA and NB rulers in one classifier will likely hurt accuracy.

2. **Sîn-šarru-iškun:** The paper includes him (21 texts = 3,133 words) but notes his corpus
   is marginal (§ 3.2). His representation is "dispersed all along the diagram." This matches
   our min-class-size concern.

3. **Class size mismatch:** Wasserman & Ni work with raw text counts per king (75–266 texts),
   not equalized subsets for probing. They use MCS to equalize for visualization, not for
   train/test balanced sampling. Our round-2 plan should implement per-class undersampling
   to n=21 (or bootstrap to n=21) for fair probe evaluation, separate from any MCS
   visualization step.

4. **No Nabonidus in RINA:** He appears only in the NBRI analysis (§ 3.3), where the
   cross-dynasty pattern is less interpretable.

**Flag for Nathan:** Our 8-ruler setup crosses the NA/NB divide. Wasserman & Ni treat these as
separate problems. We should ask whether he recommends (a) restricting to 6 NA rulers only,
(b) including NB rulers with a caveat, or (c) treating the 2 traditions as separate probes.

---

## 7. Adaptation Plan for ORCC Round 2

Based on the above extraction, the closest faithful adaptation of Wasserman & Ni's method
for our linear probing context is:

1. **Undersampling to min-class-size (n=21)** to replace MCS: draw T random subsamples
   (T=100–500) of size 21 per class, run TF-IDF + linear probe on each, report mean ±
   std of macro-F1 across draws. (MCS word-level augmentation is for UMAP visualization
   robustness; for a linear classifier, per-class undersampling is the correct analog.)

2. **IDF from full corpus:** fit the IDF on the full ORCC labeled corpus (all 893 labeled
   texts), then compute TF per virtual subset. This mirrors their "IDF from original corpus"
   rule.

3. **Evaluation:** 5-fold stratified CV within each subsample. Report mean macro-F1 and std.

4. **Phase 0 gate:** TF-IDF linear probe should reach **≥ 0.75 macro-accuracy** (matching
   their RINA 0.747 baseline) on the balanced 8-class ORCC subset to proceed to embedding
   probing.
