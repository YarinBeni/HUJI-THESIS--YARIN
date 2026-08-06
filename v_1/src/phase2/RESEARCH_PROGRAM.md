# Phase 2 — Why the Entity→Document Cliff, and What Would Fix It

A research program built from four literature sweeps (mechanistic interpretability;
information theory; representation geometry; causal inference & statistics — ~140
searches total, citations at the end of each section referenced inline) layered on top
of the first-draft diagnostics (D1–D6) and fixes (F1–F5). Written to be executable on
this repo's data: ~1,100 fragments, 8 rulers, 17 distinct years, d = 2048–8192 frozen
activations, random-init twins, parallel Akkadian/English texts, TF-IDF floor.

**The question, sharpened.** Linear probes read the year from entity-name activations
(ρ ≈ .88 famous, ≈ .5 obscure) and collapse to the twin/n-gram floor on pooled
whole-document activations. Decide between:

- **H-dilute** — the signal lives in a few tokens; mean over T tokens shrinks its SNR
  by ~k/T. *Sub-case of "present but destroyed by pooling."*
- **H-rotate** — the signal is present per-token but in token-dependent directions or
  cross-token configurations; any single pooled vector cancels it.
- **H-nonlinear** — present in the pooled vector but on a curved 1-D structure no
  linear (or low-rank PLS) read-out sees.
- **H-absent** — the model never builds a document-level "when was this written"
  variable, because next-token prediction never needed one for undated text.
- **H-artifact** — anisotropy / rogue dimensions / attention-sink contamination of
  mean-pooled decoder states masks a signal that is actually there.
- **H-OOD** — transliterated Akkadian is processed as near-garbage; the trained model
  adds little over its random twin for this input (already supported: several trained
  arms score *below* their twins in cell C).
- **H-label** — the target itself (17 values, assigned by scholars partly from the
  same textual cues) is too noisy/circular to support any ρ at document level.

These are not exclusive; the program is designed so each experiment eliminates or
quantifies specific members of this list.

**Two structural facts that govern all statistics here.**
1. The effective sample size is between 8 (rulers) and 17 (distinct years), not 1,100.
   Every test below uses ruler-level (block) permutation or wild-cluster bootstrap;
   fragment-level p-values are never reported alone.
2. I(X; year) ≤ H(year) ≤ log₂17 ≈ 4.1 bits. The right estimator family is
   classifier-based H(Y) − H(Y|X) (V-information / MDL), which is exactly feasible at
   this n. Neural MI estimators (MINE etc.) are known-bad here and are excluded.

---

## Tier 0 — Hygiene that must precede everything (days, CPU, could dissolve the result)

### T0.1 Rogue-dimension / anisotropy audit and correction  → tests H-artifact
Mean-pooled decoder states are dominated by a handful of massive-activation dimensions
(documented for Llama-2: dims 1415/2533) and attention-sink tokens; 1–3 dimensions can
carry most of cosine similarity (Timkey & van Schijndel EMNLP'21; Sun et al. 2024;
StreamingLLM ICLR'24). Diagnose per layer/condition: top-5-dimension share of cosine
similarity, participation ratio, variance spectrum. Then re-probe after each fix,
fit-on-train-only: (a) mean-centering; (b) per-dimension z-scoring; (c) all-but-the-top
with D ∈ {1,3,5,10} (Mu & Viswanath ICLR'18); (d) shrinkage-whitening to ~200 PCs;
(e) drop BOS/sink tokens and massive dims *before* pooling.
**Read-out:** probe revival under any fix ⇒ H-artifact confirmed — a methodological
finding worth its own section. No revival ⇒ every later number is reported
post-standardization and the collapse is real.
**Pitfall:** the year may live *in* a top PC (ruler vocabulary); sweep D and report the
curve, never one choice.

### T0.2 Per-layer, not best-layer — with honest reporting  → removes a standing bias
Mid-depth layers carry more probe-accessible semantics than final layers (Skean et al.
ICML'25). Every Tier-1+ quantity is computed per layer. The grid
(model × layer × pooling × probe) is then reported through (a) a hierarchical
partial-pooling model (Gelman 2012; Benavoli JMLR'17 for CV-dependent folds) and (b) a
specification curve with a permutation envelope — never the argmax cell. Best-layer
selection, where unavoidable, is done on a ruler-disjoint split (selection rulers ≠
evaluation rulers), which de-biases the winner's curse at the cost of variance.

---

## Tier 1 — Cheap and decisive (each ≤ 1 CPU/GPU-day; run all)

### T1.1 Token-level probing + late fusion (upgraded D1/F1)  → H-dilute vs H-rotate vs H-absent
Three read-outs on identical data: (i) early fusion = probe on mean-pooled vector (the
collapsed result); (ii) **late fusion** = per-token probe, aggregate *predictions*
(trimmed mean / max / noisy-OR); (iii) learned single-head attention pooling.
This is the assumption-free substitute for a formal synergy analysis, and the highest
information-per-FLOP experiment available.
**Decision rule:** (ii) or (iii) ≫ (i) ⇒ per-token present, pooling-destroyed. (ii)
fails but (iii) works ⇒ configuration/rotation. All fail while name tokens succeed ⇒
H-absent gains. Pair (iii) with the random twin and MDL costing so extra capacity
can't masquerade as signal.

### T1.2 Name-direction transfer probing  → H-dilute detector, zero new capacity
Take the year direction fitted on cell-A entity names (it exists and works), apply it
*frozen* to pooled document vectors — zero parameters learned on documents, so no
overfitting objection at n=1,000. Transfer at reduced margin ⇒ same linear variable
present but SNR-limited (dilution). No transfer while late fusion works ⇒ rotation.
Try Euclidean and whitened inner products (Park et al. ICML'24: the right inner
product may be non-Euclidean).

### T1.3 Split B′ by name presence (D3, unchanged but now with the right test)
47% of eng_tier0 fragments contain the ruler's name, 0% of akk_maximal. Split B′,
re-probe both halves under ruler-grouped MC. If the with-name half carries the whole
B′ advantage, the true cliff is at B and "entity→document" needs restating as
"name-present → name-absent." Report with wild-cluster intervals.

### T1.4 Placebo targets  → the leakage detector (new; from causal-inference sweep)
Probe the same activations for variables that **cannot** be in an ancient scribe's
text: excavation year, museum accession number, edition/publication year, corpus
record ID. Decodability above the twin floor ⇒ the pipeline reads modern
edition/formatting artifacts, and the "year" signal inherits the same suspicion;
chance-level placebos while year decodes ⇒ strong evidence against circularity.
(Choose placebos whose correlation with true date is boundable — accession numbers
sometimes correlate with excavation era.)

### T1.5 Conditional probing vs the floor + MDL everywhere  → turns nulls into numbers
Does the LLM add *any* usable bits over char-n-grams? Probe [TF-IDF ⊕ LLM] vs
[TF-IDF ⊕ 0] (Hewitt et al. EMNLP'21 conditional probing) and report **MDL online
codelength** (Voita & Titov EMNLP'20) instead of bare ρ for every cell — MDL was
designed for exactly this small-n "is the probe fooling us" regime and separates
trained from random representations where accuracy cannot. The V-information ladder
(linear → MLP → kNN; Xu et al. ICLR'20) turns "kernel probes don't help" into the
principled claim "I_V ≈ 0 for every tested V," which is the strongest absence
statement finite data can license.

### T1.6 LEACE ruler-erasure at entity level  → what does the entity result even mean
Erase ruler *identity* from cell-B representations (LEACE, Belrose NeurIPS'23,
closed-form) and re-probe year. Collapse ⇒ the entity-level ρ was an identity lookup,
not a time axis — which reframes the cliff as "documents lack recoverable entity
identity" and changes what Phase-2 should try to fix. Survival ⇒ a genuine time
direction exists independent of ruler identity.

### T1.7 OOD quantification, then stratify everything by it  → H-OOD
Per-fragment perplexity profiles (background-shift detection; Arora et al. EMNLP'21)
and relative Mahalanobis distance on activations (Ren et al. 2021). Two uses:
(a) formally establish how deep in the tail Akkadian transliteration sits;
(b) re-plot the cliff *conditional on OOD stratum* — if entity names sit near the
typical set (proper nouns occur in pretraining) while Akkadian prose is deep OOD, one
plot mechanistically explains the headline phenomenon.

---

## Tier 2 — Label-free structure (the answer to "I don't trust the labels")

### T2.1 Spectral seriation of the activation cloud  → the thesis centerpiece candidate
Archaeology's own problem — recover a 1-D chronological ordering with no labels — has
an exact spectral solution (Fiedler vector of the similarity-graph Laplacian; Atkins
et al. SIAM J. Comp. 1998; robust convex 2-SUM: Fogel NeurIPS'13; horseshoe signature
of latent 1-D orders: Diaconis et al. AoAS 2008). Recipe: kNN graph on standardized
pooled activations → Fiedler vector → candidate ordering; year labels used **once**,
post hoc, to score it (Kendall τ at document level and exact 8!-permutation p at ruler
level); run identically on the random twin to price in surface-statistics leakage.
**This defuses label circularity entirely: the estimator never sees a year.**
Also check the first diffusion coordinates for the horseshoe. Hyperparameters
pre-registered or fixed on the English side — never tuned against year.

### T2.2 Cross-model agreement without labels
Independent architectures (Llama/Qwen/OLMo/encoders) each produce an unsupervised
seriation (T2.1) and a fragment×fragment RDM. If their orderings/RDM structure agree
with *each other* above the twin-agreement floor, a shared latent axis exists no
biased label could have injected (RSA: Kriegeskorte 2008; agreement logic: Baek et al.
NeurIPS'22). Anchor against placebo axes from T1.4: date-agreement must exceed
excavation-year-agreement.

### T2.3 Gromov–Wasserstein between the Akkadian and English clouds of the *same* fragments
The parallel corpus makes this unusually well-posed (Alvarez-Melis & Jaakkola
EMNLP'18; POT's entropic GW; n=1,000 ⇒ minutes on CPU). Self-matching accuracy of the
coupling: ≫ chance ⇒ the model represents fragment content relationally in *both*
languages even though year is not linearly decodable (kills H-OOD, points at
H-nonlinear/H-rotate); ≈ twin's accuracy ⇒ correspondence is surface statistics
(supports H-OOD). If matching is high but the transported year-order τ ≈ 0, the shared
geometry is organized by genre/formula, not time — an interpretable negative.
Sweep ε with restarts; check matching isn't predicted by token count alone.

### T2.4 Geometry fingerprints: ID profiles + trained-vs-random CKA (2×2 grid)
Per layer, for {trained, twin} × {Akkadian, English}: TwoNN/GRIDE intrinsic-dimension
curves (Facco 2017; Denti 2022; the Ansuini NeurIPS'19 "hunchback" as the
in-distribution reference) and debiased CKA + Procrustes + RSA triangulated (Kornblith
ICML'19; Davari ICLR'23 for why one measure alone is untrustworthy). Read-outs: no
mid-network ID compression on Akkadian, or trained≈twin CKA through all layers ⇒
H-OOD localized to a depth; ID plateau of 2–5 at coarse scale is *necessary* for any
1-D chronological ribbon (H-nonlinear) — ID ≥ 15–20 with no plateau argues H-absent.
Seconds per layer; mandatory background for everything else.

### T2.5 Relative chronology instead of absolute years (Bradley–Terry)  → target repair
Assyriologists are far more certain that A predates B than about A's absolute year.
Re-pose the task as pairwise "older-than" and fit a Bradley–Terry/Thurstone latent
scale (bpcs; minimax design guidance: Shah et al. JMLR'16 — ensure every ruler pair is
directly compared). If a BT scale from relative judgments recovers the order while
absolute-year regression fails, the *information existed and the label was the weak
link* (H-label) — a thesis-defining dissociation, and simultaneously the "solution
model": an ordering engine rather than a year regressor.

---

## Tier 3 — Mechanistic (GPU days; run after Tiers 0–2 narrow the field)

### T3.1 Name-restoration patching  → does entity info ever reach the pooled vector
Causal-tracing variant fitted to our setup: run the name-stripped document; patch the
ruler-name token activations (from the unstripped run) back in at each layer; measure
whether the *frozen document probe* recovers. Directly tests whether the entity
pathway (Geva et al. EMNLP'23 subject-enrichment) is severed by stripping or by
pooling. Metric caveats per Zhang & Nanda ICLR'24. Forward passes only.

### T3.2 Attention knockout + head ranking at the pooled position
Knock out attention from the final/pooled positions to candidate date-bearing tokens
(eponyms, month names, formulaic openings) and rank heads by effect on the probe
read-out; check the Temporal Heads (ACL'25) and retrieval-heads (ICLR'25) candidates
first. In the entity condition, knocking out subject-token attention should reproduce
the known recall pipeline — a positive control for the method itself.

### T3.3 Patchscopes on the pooled state  → "can the model itself read it?"
Patch the last-token document representation into an inspection prompt ("This text was
written during the reign of →") in a clean context (Ghandeharioun ICML'24). The model
verbalizes reign content that probes miss ⇒ present-but-unreadable; silence converges
with H-absent. Known decoding biases apply; treat as one vote, not a verdict.

### T3.4 DAS subspace search  → last word on H-nonlinear/H-rotate
Learn a rank-≤16 rotated subspace of mid-layer *token-level* residuals such that
interchange interventions swap the probe read-out between reigns (Geiger CLeaR'24,
pyvene). Success where pooled linear probes fail ⇒ present but pooled/rotated out of
linear reach. Must run against the twin (DAS can "find" structure in random nets) and
acknowledge the expressivity critique (2025 non-linear representation dilemma).

### T3.5 SAE feature scan (discovery only, not quantification)
Qwen3-8B is covered by an open SAE suite; run the corpus through it, list features
whose document-mean activation separates reigns, and inspect what fires on date-cue
tokens. A total absence of separating features is qualitatively stronger H-absent
evidence than a failed probe — but SAE probes don't beat simple baselines for known
concepts (Kantamneni ICML'25), and English-web SAEs may lack features for this
register (a null is weak). Training our own SAE is out of budget and would be
undertrained; don't.

---

## Statistics backbone (applies to every experiment above)

1. **Ruler-level Freedman–Lane permutation of Δ(trained − twin) after residualizing
   genre/find-spot/length/preservation** — the single test that asks "date signal
   beyond confounds and surface statistics," exactly valid at the true n. Report both
   controlled and uncontrolled versions (over-controlling can delete real signal when
   preservation is itself caused by date).
2. **Wild-cluster bootstrap** (8 clusters — report as descriptive; Canay et al. 2021
   caveat) for every interval; never fragment-level bootstrap.
3. **Hierarchical model + specification curve** over the whole results grid; split-
   sample best-layer selection when a single number is needed.
4. **Quantitative bias analysis for the labels** (Lash/Fox QBA; SIMEX as sensitivity
   over a reliability grid; E-value-style inversion): report "a labeling bias of at
   least X would be required to explain the collapse." Estimate label reliability from
   inter-edition disagreement where multiple scholarly datings exist (Dawid–Skene /
   cultural-consensus on the raters). Note the honest caveat: label error here is
   differential (correlated with the text), so classical attenuation formulas bound
   nothing by themselves — the simulation grid is the defensible tool.
5. **Twins formalized as negative-control exposures**, TF-IDF as a second,
   interpretable one; Δ over twin is the only reportable effect. (Proximal-inference
   citations available for the rigorous endpoint.)

## Decision matrix (compressed)

| Pattern | Verdict |
|---|---|
| Fix in T0.1 revives probe | H-artifact — methodological headline |
| Late fusion ≫ early fusion | H-dilute/H-rotate; attention-pool is the fix |
| Name-direction transfers at reduced margin | H-dilute specifically |
| Seriation finds order, linear probes don't | H-nonlinear; and label-free chronology exists |
| GW self-matching high, year-transport ≈ 0 | shared geometry organized by genre, not time |
| ID no-plateau + CKA trained≈twin + GW ≈ twin | H-OOD; Akkadian processed as surface |
| Every V-family ≈ 0, HSIC null, late fusion fails, seriation null | best-achievable evidence for H-absent (stated as "undetectable at n=1000," never "proven absent") |
| BT relative scale works, absolute regression doesn't | H-label; ship the ordering model |

## The "solve it" endpoint (what Phase 2 ships if diagnostics permit)

- If H-dilute/H-rotate: **an unsupervised-pooling dating pipeline** — token-level
  probing + learned attention pooling, or echo-embedding (repeat the document; Springer
  2024) — evaluated under the same grouped protocol.
- If H-nonlinear: **the seriation engine itself is the product** — a label-free
  relative chronology from activations, validated post hoc, packaged with BT-scale
  uncertainty. This is also the archaeologically publishable artifact.
- If H-OOD: the fix is representational, not probe-side — the cuneiform-400M PLS
  signal says *translation supervision builds document-level variables where NTP does
  not*; the concrete test is a small encoder finetuned with a translation or
  contrastive doc-level objective (not NTP — already shown useless) and probed
  identically.
- If H-absent everywhere: the thesis claim stands as a clean boundary condition, now
  buttressed by information-theoretic, geometric, and causal evidence rather than
  probe accuracy alone — publishable as-is.

## Suggested execution order

Week 1: T0.1, T0.2, T1.1, T1.2, T1.3 (all CPU except token re-extraction).
Week 2: T1.4–T1.7, T2.4.
Week 3: T2.1, T2.2, T2.3 (the label-free block), T2.5 design.
Week 4+: Tier 3 selectively, guided by the surviving hypotheses.
