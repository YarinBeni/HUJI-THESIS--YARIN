# Phase 2 — decided experiments (v2, after the refinement round)

This supersedes the experiment list in `RESEARCH_PROGRAM.md` where they conflict.
It follows a verification round (2026-08-07) that fetched and digested the two papers
the user brought, verified what Qwen3 SAEs actually exist, mapped the Ravfogel
erasure line onto our data, and recomputed the relevant corpus facts.

---

## 0. What the refinement round established (facts, not plans)

**F1 — the corpus is bigger than the probing design ever used.** The full fragment
corpus is 1,202 fragments across **41 rulers** and **47 distinct years**. The balanced
r8 design uses only the 8 biggest rulers (1,076 fragments, 17 years). Pairwise
"which is earlier": **721,801 total pairs, 628,454 with a defined order** (different
years), 622,978 crossing rulers. The 33 long-tail rulers (1–15 fragments each),
useless for balanced regression draws, become usable as pair members.

**F2 — year is (almost) a function of ruler.** ICC(year | ruler) = 1.000 in r8.
Exactly **one** ruler has within-ruler year variance: **Esarhaddon — 176 fragments,
11 distinct years, 681–669 BCE**. Every other ruler carries a single year label.
Consequence: after a perfect linear erasure of ruler identity, the maximum linear
year signal is bounded by the within-ruler variance share ≈ 0.06%. The experiment
"erase ruler identity, see if year survives" is therefore **vacuous as stated** —
a null is guaranteed a priori, and a positive result would indicate leakage, not time.
(Verified against `akk_data.py` on the live corpus by the research agent, and
independently via the parquet here.)

**F3 — arXiv 2410.13194 is the paper we hoped it was.** "The Geometry of Numerical
Reasoning" (El-Shangiti et al., NAACL 2025 short; code:
`github.com/ahmedoumar/The_Geometry_of_numerical_reasoning`). It fits **PLS on
entity-name-token activations** to predict birth/death year (R² > .8 with 5
components), then does the causal step: take **w = pls.x_weights_[:, 0] ·
scaler.scale_**, hook one layer, add `(α/‖w‖)·w` at the last token of the second
entity in "Was X born prior to Y? Output only Yes or No.", sweep α ∈ arange(−30,
30, 1.5) — and the Yes/No answers **flip**, far above an equal-norm random-direction
control, with effects concentrated in the **first half of the layers**. Their split
is a random row split (no entity holdout) — our GroupKFold protocol is strictly
stricter, which is worth a citation-plus-contrast paragraph.

**F4 — official Qwen3 SAEs exist (Qwen-Scope, ~May 2026).** Residual-stream TopK
SAEs on HuggingFace: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_{50,100}` (all 28
layers) and `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_{50,100}` (all 36 layers, d_in
4096, width 65,536). Hook point = post-block residual stream — **exactly what our
extraction saves** (`output_hidden_states[1:]`), so our cached activations can be
pushed through the SAE offline with plain torch (one `layer{n}.sae.pt` per layer),
or via `sae-lens ≥ 6.43.0`. Two mismatches to manage: (a) the SAEs are trained on
the **-Base** checkpoints while our ladder extracted the post-trained Qwen3 models —
measure FVU on our acts first, or cheaply re-extract with Base; (b) **no Qwen3-32B
coverage** (one community SAE at layer 32 only, Neuronpedia
`qwen3-32b/32-resid-batchtopk-65k`). Auto-interp feature descriptions: the official
Scope features are browsable in the `Qwen/QwenScope` HF Space; Neuronpedia's
auto-interp covers the mwhanna **MLP transcoders** for qwen3-1.7b/8b (a different
object — transcoder features, not residual SAE features) — plus whatever metadata
we obtained from Chongrong and Nathan.

**F5 — the Ravfogel line, precisely.** LEACE (NeurIPS 2023, pip
`concept-erasure`): closed-form affine map that zeroes the cross-covariance
Cov(X, Z), provably defeating *every* linear predictor of Z under *every* convex
loss, with the minimum-norm change to X; rank ≤ k−1 for a k-class concept. It
removes **first-moment linear** information only — nonlinear/covariance structure
survives, so every "erased" claim needs a nonlinear check. RLACE (ICML 2022):
rank-budgeted adversarial erasure; showed rank-1 often suffices — its directions
double as interpretable steering vectors. The unlearning-traces paper (EMNLP 2025,
ConceptVectors): behavioural unlearning leaves "parametric knowledge traces" —
concept-specific MLP value vectors — almost untouched and recoverable; so removal
claims must be tested **intrinsically** (inside the weights/activations), not
behaviourally. The trace tool that transfers directly to us: **logit-lens the probe
direction itself** and look at which tokens dominate its vocabulary projection.

**F6 — arXiv 2602.15730** ("Causal Effect Estimation with Latent Textual
Treatments", Feldman, Venugopal, Spiess, Feder — Feb 2026): selects SAE features as
*treatments*, steers generation to create quasi-counterfactual texts, and — the key
statistical point — shows **steering is not surgical**: it drags correlated nuisance
features along, and naively adjusting for raw embeddings is biased *because the
embedding contains the treatment*. Their fix is covariate **residualization**
(partial the treatment out of each embedding coordinate, or drop the first PC). For
us this upgrades every "suppress the time feature — does the prediction flip?" test
with a mandatory side-effects audit. (Caveat: both paper digests were reconstructed
from search-snippet quotations because arXiv is egress-blocked in this environment;
pull the PDFs before citing specifics in the thesis.)

---

## 1. The decided experiments

### E1 — Pairwise chronology: "which text is earlier?" (the dataset-expansion move)

**Context.** The regression design throws away 33 of 41 rulers because they cannot
fill a balanced draw. Recast dating as pairwise comparison and they all come back:
628k ordered pairs instead of ~1.1k labeled fragments, built from C(1202, 2).
Bradley–Terry-style relative chronology is also *less* exposed to absolute-label
bias — a pair's order survives a systematic shift in the labels.

**What we do.** Two versions, sharing pair generation:
1. *Representation-side*: a pairwise probe on activation differences
   (score(x_i) − score(x_j) → logistic "i earlier than j"), trained under
   **group-aware folds where both rulers of a test pair are unseen in training**.
   Arms: the usual ladder + TF-IDF floor + random twins.
2. *Behavioural*: adopt the NAACL paper's protocol nearly verbatim — balanced
   Yes/No pairs, "Was inscription A written before inscription B?", P(A,B) vs
   P(B,A) consistency check, two-shot variants, single-token Yes/No scoring —
   on English glosses (cell B′) and raw Akkadian (cell C).

**Honest accounting.** The 628k pairs are training fodder, not independent
evidence: the independent units are **ruler pairs — C(41,2) = 820** — and test
metrics (pairwise AUC) are reported at ruler-pair level with cluster-aware
uncertainty (§3). Within-ruler pairs are unordered (same year) except within
Esarhaddon — those go to E6.

**Read-out.** If trained models beat the TF-IDF floor at *relative* ordering where
they failed at absolute regression, document-level time exists but the absolute
calibration was the problem. If the floor still wins, the collapse is deeper than
the task format.

### E2 — Causal steering along the fitted PLS direction (the NAACL recipe)

**Context.** We already fit PLS probes on cell-A entity activations; the NAACL
paper shows the first PLS component is not just decodable but **used**: adding it
at the entity's name token flips the model's earlier/later answers. That is the
causal upgrade of our cell A, and a published recipe we can run as-is. This is the
experiment the user flagged as most appealing ("מאוד קורץ").

**What we do.** Ladder of three, same intervention each time
(w = x_weights[:,0]·scale; add (α/‖w‖)·w at the name token; sweep α and layers in
the first half of the stack; controls: equal-norm random direction **and our
random-init twins**, a control the paper lacks):
1. *Cell A replication* on Llama-2-7B / OLMo-2-7B / Qwen3-8B: do Yes/No
   birth-order answers flip? (Expected: yes — validates the harness.)
2. *Cell B*: same with Assyrian ruler names in English. Flip rate vs cell A
   localizes how much of the B degradation is causal-use vs read-out.
3. *Cell C*: steer at ruler-name tokens inside Akkadian fragments while asking
   earlier/later. **If English names flip and Akkadian fragments don't, that is
   direct causal evidence the collapse happens because no name-token anchor
   engages the comparison circuit** — the mechanism-level version of our thesis
   claim.

**Read-out.** Flip-rate curves (dose-response in α) per cell per layer band.

### E3 — Frozen name-direction transfer + the LEACE mediation test (merged, reframed)

**Context.** Original question: is document time the *same axis* as entity time,
just weaker — or a different axis? And the erasure question had to be reframed:
by F2, "erase ruler → does year survive" is vacuous. The well-posed version tests
**mediation**: does the transferred name direction order fragments *through ruler
identity*, or independently of it?

**What we do.**
1. Freeze the cell-A year direction (ridge and PLS variants). Score all r8
   fragment activations with it — zero document-side fitting. Read Spearman
   against chronology.
2. LEACE-erase one-hot ruler identity (fit on train folds only,
   `concept-erasure`, shrinkage on — d=4096 ≫ n≈1071), re-score with the same
   frozen direction, compare orderings before/after.

**Read-out.** Transfer works + survives erasure → a ruler-independent time
component exists in document representations (strong, label-light result).
Transfer works but collapses under erasure → the transfer was an identity lookup —
which *is* the answer to "why the collapse". Transfer fails outright → entity time
and document time are different axes.

### E4 — Confounder erasure + surgical controls + trace inspection

**Context.** The user's original interest — "a model from which we linearly delete
the relation to certain information" — pointed at confounders, and that use is
well-posed (year varies within genre/length/provenance classes). Plus the traces
paper says: never claim "erased" or "survived" from a linear probe alone.

**What we do.**
1. *Confounder erasure*: Z = [genre one-hot; log-length + quantile-bin
   indicators; provenance]. LEACE per train fold; re-run the cell-C probes; does
   the nested-k PLS gap (.336 vs .243) survive? Apply the same erasure to the
   TF-IDF floor so it moves consistently. Caveat to decompose, not hide:
   provenance predicts ruler at 55.6% and median length varies by ruler (88–183
   chars), so confounder erasure partially erases ruler/year through correlation —
   report ruler decodability before/after alongside.
2. *Surgical controls*: (a) LEACE(year) — a rank-1 nick — then verify ruler
   classification is intact; (b) after LEACE(ruler), verify a linear ruler probe
   on held-out fragments falls into the permutation null band.
3. *Intrinsic battery*: quadratic + small-MLP + RBF ruler probes after each
   erasure (reusing the P10 reduce-then-kernel code); any above-chance nonlinear
   recovery is reported next to every claim.
4. *Trace inspection, no erasure needed and immune to the F2 degeneracy*:
   **logit-lens the ridge/PLS year direction** in OLMo-2-7B / Llama-2-7B. If its
   vocabulary projection is dominated by royal names and toponyms rather than
   temporal vocabulary, that is intrinsic evidence the cell-C "year" signal is
   identity/name-mediated — publishable on its own.
5. *Optional RLACE*: rank-1..4 adversarial ruler subspaces; the minimal identity
   directions double as steering vectors for E2/E5.

### E5 — SAE track on Qwen3 (Qwen-Scope), gated and audited

**Context.** Official SAEs exist for exactly two of our arms (F4), our cached
activations are the right hook point, and the user has feature metadata from
Chongrong and Nathan. The 2602.15730 paper supplies the discipline: feature
suppression is not surgical, so causal claims need a side-effects audit.

**What we do**, in order, each step gating the next:
1. *FVU gate*: push cached qwen3_8b activations through `layer{L}.sae.pt` for
   L ∈ {8, 16, 20, 24, 28}; report FVU per layer for English entities vs Akkadian
   fragments. This measures both the base-vs-post-trained checkpoint mismatch and
   the OOD question ("what happens to SAE features on Akkadian?") in one number.
   If FVU is bad on the post-trained acts, re-extract with Qwen3-8B-Base (cheap;
   arguably the better arm for a pretraining-time claim anyway).
2. *Feature hunt (cell A)*: TopK feature activations at the probe's best layer;
   rank by Spearman with year and by cosine(W_dec row, our PLS year direction);
   inspect top ~50 in the QwenScope Space / transcoder Neuronpedia pages / the
   Chongrong-Nathan metadata. Deliverable: a small dictionary of "time features".
3. *Collapse at feature level*: do those year features fire on cell B names and
   on Akkadian fragments at all? A null is itself a phase-2 result: document
   time never reaches the feature basis.
4. *Counterfactual suppression with the residualization discipline*: suppress or
   clamp candidate time features, read the frozen year probe as dose-response —
   while **simultaneously tracking a ruler-identity probe and TF-IDF
   reconstruction error** (the non-surgicality audit), and comparing naive vs
   residualized effect estimates per 2602.15730.
5. *W_dec rows as steering vectors* — cross-validated against E2's PLS direction:
   two independently-derived "time directions" agreeing is strong convergent
   evidence.

**Scope note.** qwen3_32b has no usable SAE coverage (single layer 32) — it sits
out of this track or gets the one-layer treatment only.

### E6 — The Esarhaddon micro-study (new; forced by F2)

**Context.** Esarhaddon is the **only** place in the corpus where "document-level
time independent of ruler identity" is even defined: 176 fragments, 11 distinct
years, a 12-year window. Everything else confounds year with identity by
construction.

**What we do.** Within his fragments only: linear/PLS year probes vs the TF-IDF
floor and random twins; within-ruler pairwise ordering (the E1 machinery); Fiedler
seriation restricted to his fragments; significance by permuting year-group labels
within Esarhaddon. Optionally LEACE(length/genre) inside the subset.

**Read-out.** Any within-Esarhaddon signal is identity-free by construction —
small n, but the cleanest single claim available. A null is honest too: at ±2.5y
label sd over a 12-year window, we state the detection floor.

### E7 — Spectral seriation (kept as decided last round)

Unsupervised Fiedler ordering of the fragment kNN graph (reusing the p9 geodesic
graph code), labels used once post hoc. Unchanged; now also runs within-Esarhaddon
(E6) and after confounder erasure (E4) as cheap add-ons. Connection noted in
2602.15730's "drop the first PC" residualization: if the dominant unsupervised
direction is identity/register rather than time, seriation will say so directly.

### E8 — Statistics backbone (applies to all of the above)

- **Ruler-level Freedman–Lane permutation**: permute ruler-year assignments
  (8! in r8; larger in r41), residualizing genre/length first. The only honest
  null under ICC = 1.
- **Wild-cluster bootstrap** for CIs (clusters = rulers; effective n = 8 or 41,
  never 1,202).
- **Pairwise metrics at ruler-pair level** (820 independent units, not 628k).
- **Specification curve** over layer × cleaning × pooling × probe for each
  headline claim.
- **QBA / E-value** sensitivity: how wrong would the year labels have to be to
  erase the effect.
- 2602.15730's **naive-vs-residualized gap** reported for every steering claim.

---

## 2. Dropped or reframed, and why

| was | became | why |
|---|---|---|
| "LEACE ruler identity → does year survive?" | E3 mediation test + E6 micro-study | F2: ICC(year\|ruler)=1.000 makes the original vacuous — null guaranteed, positive = leakage |
| generic contrast-pair steering | E2 with the fitted PLS x-weight direction | F3 gives a published, causally-validated recipe; our probes already produce w |
| "SAE if available" | E5 with FVU gate + suppression audit | F4: Qwen-Scope verified real; F6: suppression without a side-effects audit is not evidence |
| BT as an aside | E1 as a first-class experiment | F1: 628k ordered pairs, 41 rulers, long-tail rulers re-enter — the user's dataset-expansion point checks out |

## 3. Suggested order of execution

1. **E1** pairwise (pure CPU on cached activations; biggest dataset win first)
   + **E4.4** logit-lens traces (free, no GPU, immediately interpretable).
2. **E3** transfer + mediation (CPU, needs only `concept-erasure` + cached acts).
3. **E5.1** FVU gate (CPU) → decides the SAE track's shape.
4. **E2** steering (GPU, forward hooks, small models first).
5. **E4** erasure suite + **E6** Esarhaddon + **E7** seriation.
6. **E8** wraps every result as it lands.

## 4. Verification caveats

The two arXiv digests (F3's paper partially — its code repo was read in full — and
F6's paper especially) were reconstructed with arXiv egress-blocked; F6's exact
algebra is unverified. Before any of this enters the thesis text, pull the PDFs:
2410.13194 (NAACL 2025), 2602.15730, 2306.03819 (LEACE), 2201.12091 (RLACE),
2406.11614 (ConceptVectors), 2605.11887 (Qwen-Scope).
