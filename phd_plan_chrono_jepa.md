# PhD Plan: Chrono‑JEPA / Chrono‑Barlow

## Self‑Supervised, Confound‑Resistant Chronological Representation Learning for Low‑Resource Ancient Languages — with Sparse Interpretability

**Builds on:** M.Sc. "Boundary Conditions on Emergent World Models" (Beer, 2026) — which established that (a) NTP LLMs at any scale carry no recoverable diachronic structure for Akkadian text (ρ ≈ random floor at mean‑pool), (b) declarative knowledge and linear representation are separable, and (c) the **multilingual translation objective** (Thalesian‑cunei‑400M) is the lone lever that installs learned chronological structure (0.897 F1 / ρ = 0.851 maxking; deepening signal across layers, peaking L10–L11).

---

# PART I — THE PhD THESIS PLAN

## 1. One‑paragraph thesis statement

> The M.Sc. showed *where* chronological structure lives (translation‑trained cuneiform encoders) and *where it does not* (NTP LLMs of any scale). The PhD asks how to **create** that structure deliberately and safely: a self‑supervised objective — a JEPA‑style predictive twin with a Barlow‑Twins redundancy‑reduction loss whose *augmentations are confound removals* and whose ordinal axis is trained with a differentiable soft‑rank (soft‑Spearman) loss on **relative order only, never absolute years** — turns 1,202 labeled fragments plus 2M unlabeled words into a combinatorially large supervision signal. A sparse‑autoencoder decomposition then converts the learned time axis into auditable, philologically nameable diachronic features (ChronoAtlas), and the whole recipe is tested for generality on a second low‑resource ancient corpus.

## 2. Why SSL, why JEPA/Barlow, why not year regression

| Problem (from M.Sc.) | SSL answer |
|---|---|
| 1,202 labels, 41 rulers, heavy imbalance | Pairwise/ordinal + multi‑view construction: N texts × M augmented views × O(N²) order constraints — supervision grows combinatorially, not linearly, while the invariance term needs **no labels at all** (uses the 2M‑word corpus) |
| Probes exploit shortcuts (length, ruler token, formulae, genre) | In Barlow Twins, invariance is enforced *across augmentations*. If augmentations = **confound removals** (name masking, formula deletion, crops, orthographic normalization), then confound‑invariance is built into the representation by construction, not patched afterwards |
| Direct year regression on 1‑D target overfits and produces fake precision | The SSL head never sees a year. It sees only: "these two views are the same document" (invariance), "these embedding dims must decorrelate" (redundancy reduction), and "text i precedes text j" (soft‑rank on relative order). Absolute years enter only at the very end, through a frozen 1‑D coordinate + monotone calibrator with interval likelihood |
| Token‑identity illusion (random model wins at king token) | JEPA predicts in *representation space* from a masked/degraded view — it cannot solve the task by token‑identity lookup because identity carriers are exactly what the augmentations delete |
| Spearman ρ is the evaluation metric but is non‑differentiable | Use soft‑rank / differentiable Spearman surrogates (sigmoid pairwise, torchsort/FastSort, optimal‑transport ranks); optionally HSIC (CDSSL‑style) to kill *nonlinear* dependence on confounds |

## 3. Research questions & hypotheses

- **RQ1 (Method):** Can a self‑supervised twin objective (invariance‑to‑confound‑removal + redundancy reduction + soft‑rank ordinal axis) learn a chronological coordinate over frozen translation‑encoder representations that beats supervised PLS/ridge/ordinal heads under the maximal‑balanced protocol?
  - **H1:** Yes, and the margin *grows* under stress evaluations (leave‑genre‑out, name‑masked, formula‑removed) even if the headline ρ gain is modest (e.g., 0.41 → 0.45), because invariance is architectural.
- **RQ2 (Data leverage):** Does the label‑free invariance term over the 2M‑word unlabeled corpus improve ordering learned from only the 1,202 labeled fragments (semi‑supervised gain), and does a corpus kNN‑graph pseudo‑ordering (seriation) add further gain without introducing genre/source leakage?
  - **H2:** Unlabeled invariance + graph smoothing improves robustness metrics more than headline ρ.
- **RQ3 (Active ingredient, causal version):** The M.Sc. established translation‑objective correlationally (one checkpoint). Train **controlled mini‑encoders from scratch** — same architecture/size/data, varying only the objective (NTP vs. MLM vs. translation vs. denoising seq2seq) — to causally isolate *which property of translation training* (semantic compression? encoder bidirectionality? cross‑lingual alignment?) installs diachronic structure.
  - **H3:** Cross‑lingual seq2seq objective > monolingual denoising > MLM > NTP at equal scale/data; the deepening‑with‑layers signature is objective‑specific.
- **RQ4 (Interpretability):** Is the learned time axis decomposable via SAE into sparse features that (a) correlate with date, (b) survive causal deletion tests, (c) map onto known philological diachronic markers, and (d) surface *new* candidate markers validated by an Assyriologist?
  - **H4:** A concept‑bottleneck dater restricted to ~50 named features retains ≥ 80% of the full head's performance.
- **RQ5 (Generality / boundary conditions, part 2):** Does the recipe transfer to a second low‑resource diachronic corpus (candidates: Sumerian; dated Hittite; epigraphic Hebrew/Aramaic; and one *high‑resource* sanity domain, e.g., dated historical English, where the world‑model literature says linear probes already work)?
  - **H5:** The recipe transfers; on the high‑resource domain SSL ≈ linear probing (consistent with the boundary‑condition thesis: SSL is needed exactly where emergent structure is absent).

## 4. Core system: Chrono‑JEPA‑Barlow (CJB)

```text
                         ┌────────────────────────────┐
   view A (confound-     │  Online encoder  f_θ       │──► z_A ──► projector g ──► p_A
   masked augmentation)  │  (frozen Thalesian L11 +   │        │
                         │   trainable adapter/head)  │        └─► s(x) = wᵀh  (1-D chrono axis)
                         └────────────────────────────┘
                         ┌────────────────────────────┐
   view B (different     │  Target encoder  f_ξ       │──► z_B ──► projector g' ──► p_B
   augmentation)         │  (EMA of θ, stop-grad)     │
                         └────────────────────────────┘

   L = L_BT(p_A, p_B)                    ← cross-correlation → identity  (invariance + decorrelation)
     + λ_rank · L_softρ(s, weak order)   ← differentiable Spearman on RELATIVE order only
     + λ_var  · L_variance(s)            ← VICReg-style anti-collapse on the scalar axis
     + λ_hsic · HSIC(s, confounds)       ← nonlinear deconfounding vs. genre/length/source
     + λ_graph· Σ w_ij (s_i − s_j)²      ← kNN-graph smoothness over the 2M-word unlabeled corpus
```

**Design commitments**

1. **Backbone frozen first.** Thalesian‑cunei‑400M layer ~11 (validated best in M.Sc.). Only adapter + projector + head train. Backbone LoRA is a Phase‑2 ablation, never the starting point.
2. **Augmentations = confound removals** (the central trick):
   `<RULER>/<DIVINE>/<PLACE>/<OFFICIAL>` typed masks · formulaic opening/closing deletion · 8/16/32/64‑word crops · orthographic normalization variants (normalized vs. simple transliteration, sign‑level) · logogram/determinative stripping · random span drop. On‑diagonal BT term forces s(x) to be identical across all of them → the axis *cannot* be ruler‑name detection, length, or formula matching by construction.
3. **Order signal is weak and relative.** Sources: reign intervals → pairwise precedence with distance‑scaled margins (only when intervals are disjoint); ruler adjacency chains; eponym/synchronism constraints; high‑confidence graph seriation pairs from the unlabeled corpus. Ties/overlapping intervals contribute **no** rank gradient.
4. **Years only via calibration.** After training, freeze s(x); fit a monotone map s → p(t|x) with interval NLL + conformal prediction. Report intervals + coverage, never bare point years.
5. **JEPA element.** The target branch sees a *less degraded* view; the online branch predicts its representation from a *more degraded* view (masked names, cropped). Prediction in latent space of what the fuller document "would look like" chronologically — this is the data‑exponential engine: every (document × augmentation‑pair) is a training example.

## 5. Thesis chapters (≈ 4 years)

| Ch. | Title | Content | Target venue | Year |
|---|---|---|---|---|
| 1 | Boundary Conditions (extended M.Sc.) | Honest benchmark; failure of scale & NTP; token‑identity illusion; layerwise autopsy | (done — journal version) | Y1 |
| 2 | The Active Ingredient, Causally | Controlled from‑scratch objective ladder (RQ3); what property of translation installs time | ACL/EMNLP | Y1–Y2 |
| 3 | Chrono‑JEPA‑Barlow | Method + semi‑supervised gains + full stress‑evaluation suite (RQ1, RQ2) | NeurIPS/ICLR | Y2–Y3 |
| 4 | Feature Archaeology | SAE decomposition, causal tests, concept‑bottleneck dater, ChronoAtlas tool + expert study (RQ4) | ACL / interp venue + DH journal | Y3 |
| 5 | Generality | Second ancient corpus + high‑resource sanity domain (RQ5); boundary‑condition theory synthesis | thesis + journal | Y4 |

## 6. Evaluation contract (fixed for the whole PhD — no metric shopping)

**Primary:** maximal‑balanced Spearman ρ (200 Monte‑Carlo subsets, 8 rulers × 21 fragments, GroupKFold by ruler), exactly as in the M.Sc.
**Robustness battery (every model, every experiment):** leave‑one‑ruler‑out · leave‑one‑genre‑out · name‑masked · formula‑removed · source‑held‑out · fragment‑length curve (8/16/32/64/full) · counterfactual formula‑insertion attack · date‑shuffle placebo (must → ρ ≈ 0) · random‑init backbone control (must not match trained system).
**Calibration:** MAE, pairwise order accuracy, 80/90% interval coverage, NLL.
**Interpretability faithfulness:** evidence‑deletion vs. matched random deletion; SAE feature ablation moves prediction as predicted; feature stability across splits.
**Success definition (pre‑registered):** system counts as a win if it (i) ≥ PLS baseline on primary metric AND (ii) degrades ≤ half as much as PLS under name‑masking, formula‑removal, and leave‑genre‑out AND (iii) intervals are calibrated (coverage within ±5% of nominal). A +0.03 ρ with 2× robustness beats +0.10 ρ that collapses under masking.

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Scalar axis collapses (BT invariance kills the ordinal signal) | Med | variance term on s(x); tune λ_rank warm‑up; monitor rank‑corr on train weekly |
| Soft‑rank surrogate diverges from true Spearman | Med | unit test surrogate↔scipy agreement > 0.99 at low temperature; two independent implementations (sigmoid‑pairwise & torchsort) |
| Augmentation leakage (masking itself encodes period, e.g., mask density correlates with genre) | Med | audit: probe s(x) for mask‑count; add mask‑count adversary if leaked |
| Graph pseudo‑order imports genre/source clusters | High | build graph within‑source only for smoothing; never cross‑source pseudo‑pairs; ablate with/without |
| HSIC/adversary deletes true diachronic signal (ruler ↔ date is real) | High | deconfound only *presence/identity* nuisances, never date‑correlated content; always report the no‑adversary variant |
| From‑scratch objective ladder too expensive | Med | mini‑scale (37–100M) encoders, matched tokens; the M.Sc. 37M MLM already proves mini‑scale is informative |
| Expert time (Assyriologist validation) | Med | design Ch.4 study early; batch feature review sessions; fallback = published diachronic marker lists |
| Second‑corpus data quality (RQ5) | Med | choose corpus in Y1; reuse ORACC infrastructure (Sumerian shares it) |

## 8. What is explicitly out of scope (first 3 years)

Generative diffusion over text · NODE/NCDE trajectory models · full‑backbone finetuning before frozen‑head results · English translation as the main dating representation (kept only as a confound diagnostic) · SAE as a *performance* mechanism (discovery only) · accuracy‑only reporting.

---

# PART II — EXECUTABLE WORK PLAN (junior / agentic-coder ready)

Every ticket below has: **ID · Goal · Inputs · Steps · Deliverable · Verification (machine-checkable gate)**. A ticket is "done" only when its verification passes in CI. No ticket should take more than ~2–4 focused days; if it does, split it.

## Repo & engineering contract (Phase 0 output)

```text
chrono/
  data/            # loaders, schemas, splits (never raw data in git)
  augment/         # confound-removal augmentation engine
  losses/          # bt.py, softrank.py, hsic.py, variance.py, interval_nll.py, graph.py
  models/          # backbones.py (frozen wrappers), heads.py, jepa.py, ema.py
  eval/            # protocol.py (maximal-balanced), robustness.py, calibration.py
  interp/          # sae.py, feature_report.py, ablation.py
  atlas/           # retrieval + report rendering
  configs/         # yaml per experiment; every run = 1 config + 1 git hash + 1 seed
  tests/           # pytest; every loss and every protocol has a unit test
  scripts/         # run_experiment.py, make_splits.py, extract_embeddings.py
```

Rules for the coder/agent:
- Deterministic seeds everywhere; every result table regenerable by `python scripts/run_experiment.py configs/X.yaml --seed S`.
- All embeddings cached to disk as `.npy` + a `manifest.parquet` (doc_id, model, layer, site, sha of text version). Heads train from cache — no GPU backbone pass inside training loops.
- Every metric written to a single `results.parquet` (run_id, config_hash, metric, value, split, seed). Plots are generated from that file only.
- A `NULLS.md` invariant file: date-shuffle placebo must give ρ ∈ [−0.05, 0.05]; random-init backbone recorded for every experiment. CI fails if a run beats its own placebo by < 0.05.

---

## Phase 0 — Reproduce the M.Sc. floor (Weeks 1–4)

**P0.1 — Data contract**
- Goal: one canonical dataset table.
- Inputs: ORACC/eBL/Archibab merged corpus (1,202 fragments) + 2M-word unlabeled corpus.
- Steps: build `corpus.parquet` with columns: doc_id, transliteration_raw, transliteration_norm, ruler, reign_start, reign_end, year_point, genre, provenance, source, n_words, named_entity_spans (typed), formula_spans.
- Deliverable: `corpus.parquet` + `schema.md`.
- Verify: pytest asserts row count = 1,202; no null rulers; every reign interval start ≤ end; NE spans within text bounds; SHA-pinned.

**P0.2 — Split factory**
- Goal: all splits pre-generated and frozen.
- Steps: GroupKFold-by-ruler; 200 balanced Monte-Carlo subsets (8×21); leave-one-ruler-out; leave-one-genre-out; source-held-out. Save as JSON lists of doc_ids.
- Verify: no doc_id appears in both train and test of any split; balanced subsets have exactly 8 rulers × 21 docs; snapshot test (splits byte-identical across runs).

**P0.3 — Embedding extraction cache**
- Goal: frozen representations on disk.
- Steps: extract Thalesian layers 0–12 (mean, last, king sites), Qwen3-8B L16, TF-IDF vectors, random-init Qwen3-8B, for all text versions (raw, maximal-clean, maximal-with-kings).
- Verify: manifest completeness check; cosine self-similarity of re-extracted sample = 1.0 (determinism).

**P0.4 — Baseline reproduction (GATE)**
- Goal: reproduce M.Sc. numbers before anything new.
- Steps: PLS + ridge probes on cache; run maximal-balanced protocol.
- Verify: Thalesian L11 ρ = 0.41 ± 0.02; Qwen3-8B L16 ρ = 0.36 ± 0.02; TF-IDF ρ ≈ 0.29; shuffled-label ρ ≈ 0. **No further phase starts until this gate passes.**

---

## Phase 1 — Augmentation & confound engine (Weeks 4–8)

**P1.1 — Typed masking**
- Steps: functions `mask_rulers`, `mask_divine`, `mask_places`, `mask_officials` replacing spans with typed tokens; property-based tests (masking twice = masking once; non-NE text unchanged).
- Verify: on 50 hand-checked docs, ≥ 98% of gold NE spans masked, 0 false masks on a clean synthetic doc.

**P1.2 — Formula remover**
- Steps: rule library for openings/closings/titulary (regex + span table from P0.1); `remove_formulae(doc)`.
- Verify: expert-labeled 100-doc sample: precision ≥ 0.9 on formula spans; output never empty (min 5 words retained, else flag).

**P1.3 — Crops & orthography views**
- Steps: deterministic-by-seed 8/16/32/64-word crops; normalized/simple/sign-level transliteration converters.
- Verify: round-trip tests; length distributions match spec; every doc yields ≥ 6 distinct valid views.

**P1.4 — View sampler**
- Steps: `sample_view_pair(doc, seed) → (view_A, view_B)` with configurable augmentation menus per branch (target branch = milder).
- Verify: unit test — for fixed seed, pairs reproducible; over 10k samples, each augmentation appears within ±10% of its configured probability.

**P1.5 — Confound audit table**
- Steps: for each view type, compute length, mask counts, genre, source; store `confounds.parquet`.
- Verify: table joins 1:1 with views; used later by HSIC and leakage probes.

---

## Phase 2 — Loss library (Weeks 6–10, parallel with Phase 1)

Each loss is a standalone `nn.Module` with a pure-tensor API and its own test file.

**P2.1 — Barlow Twins loss**
- Verify: on synthetic identical views → loss ≈ off-diag term only; gradient check (`torch.autograd.gradcheck`); reproduces reference implementation loss on a fixed tensor to 1e-5.

**P2.2 — Soft-rank / soft-Spearman loss (two implementations)**
- Steps: (a) sigmoid-pairwise surrogate with distance-scaled margins; (b) torchsort/soft-sort based differentiable Spearman.
- Verify: on synthetic data with known order, minimizing loss drives true scipy Spearman > 0.99; both implementations agree within 2%; margin scaling test (far pairs get larger gradient than adjacent pairs).

**P2.3 — Variance (anti-collapse) term** — Verify: collapses to zero variance without it in a toy run; stays > threshold with it.

**P2.4 — HSIC penalty** — Verify: HSIC(x, x²) detected (nonlinear dep.) while Pearson ≈ 0; HSIC(x, independent y) ≈ 0; matches reference RBF-HSIC values.

**P2.5 — Interval NLL + monotone calibrator** — Verify: on synthetic interval data, coverage of learned intervals within ±3% of nominal; calibrator monotonicity asserted.

**P2.6 — Graph smoothness** — Verify: on a chain graph with known ordering, loss decreases as scores approach ordering; per-source-masking option covered by test.

**P2.7 — Ordered-pair generator from weak labels**
- Steps: emit (i, j, margin) only for disjoint reign intervals; balanced sampling by ruler/period/genre; cap pairs per epoch.
- Verify: zero pairs emitted for overlapping intervals; per-ruler pair counts within 2× of each other; total pairs per epoch = config value.

---

## Phase 3 — Minimal Chrono-Barlow head (Weeks 10–16) ← first science result

**P3.1 — Trainer skeleton**
- Steps: dataloader (cached embeddings + augmentation views re-embedded offline via P0.3 extension), linear head + projector, L = L_BT + λ_rank·L_softρ + λ_var; config-driven; logs to results.parquet.
- Verify: overfit test — on 50 docs, train rank-corr → > 0.95; loss curves monotone; run finishes < 30 min CPU/1-GPU.

**P3.2 — View embedding cache extension**
- Steps: pre-embed all augmented views through frozen Thalesian (offline, once).
- Verify: manifest covers every (doc × view); spot-check 10 embeddings re-extracted identically.

**P3.3 — Experiment E-MIN (GATE)**
- Steps: full maximal-balanced protocol comparing: PLS · ridge · linear ChronoRank (rank+interval only, no BT) · Chrono-Barlow (BT + rank) · each ± name-masking training views. 5 seeds.
- Deliverable: table + plots.
- Verify (gate to Phase 4): Chrono-Barlow ≥ PLS on primary ρ; degradation under name-masked eval ≤ 50% of PLS's degradation; placebo passes.

**P3.4 — Robustness battery run**
- Steps: run `eval/robustness.py` (leave-ruler-out, leave-genre-out, formula-removed, length curve, counterfactual insertion) on all P3.3 models.
- Verify: report auto-generated; every cell has 5-seed mean ± sd; CI fails on missing cells.

**P3.5 — Leakage probe**
- Steps: linear + MLP probes predicting mask-count, length, genre, source from s(x).
- Verify: report produced; if genre/length decodable at > baseline+0.1 F1, file follow-up ticket to add HSIC term (P4.2).

---

## Phase 4 — Full Chrono-JEPA + unlabeled corpus (Months 5–10)

**P4.1 — EMA target branch + latent predictor**
- Verify: EMA unit test (ξ ← mξ + (1−m)θ exact); stop-grad verified (target grads = None); training stable 100 epochs without collapse (variance monitor).

**P4.2 — HSIC deconfounding arm** — Verify: leakage probe (P3.5) drops below threshold; primary ρ drops < 0.02.

**P4.3 — Unlabeled invariance stream**
- Steps: mix 2M-word corpus docs into the BT term (no rank loss for them).
- Verify: semi-supervised ablation — with vs. without unlabeled stream, 5 seeds; result logged either way (negative result is a result).

**P4.4 — kNN graph + smoothing**
- Steps: within-source kNN graph over unlabeled corpus (Thalesian embeddings); add L_graph.
- Verify: graph stats report (degree dist, source purity = 100% by construction); ablation table with/without.

**P4.5 — Counterfactual attack hardening**
- Steps: adversarial evaluation — insert period-alien formula; measure |Δs|.
- Verify: |Δs| of CJB ≤ 50% of PLS's |Δs| on the same attack set.

**P4.6 — Paper 1 freeze (Ch. 3)**
- Verify: one command regenerates every table/figure from results.parquet at a pinned git hash.

---

## Phase 5 — Calibration + ChronoAtlas (Months 9–13)

**P5.1 — Conformal intervals over frozen s(x)** — Verify: 80/90% coverage within ±5% on held-out rulers.
**P5.2 — Retrieval module** — nearest dated parallels in chronology-aware space. Verify: retrieval-coherence metric (neighbor date MAE) beats raw-embedding retrieval; genre-share of neighbors reported (shortcut check).
**P5.3 — Report renderer** — per-fragment card: interval, confidence, parallels, earlier/later evidence, confound warnings. Verify: golden-file test on 5 fragments; historian review checklist attached.

---

## Phase 6 — Interpretability: SAE & feature archaeology (Months 12–20)

**P6.1 — SAE training on head inputs** — Verify: reconstruction R² ≥ 0.9 at L0 ≤ 64; dead-feature fraction < 20%.
**P6.2 — Feature–date screening** — per-feature date correlation + top-activating spans. Verify: auto report for top 100 features; stability ≥ 0.7 Jaccard across two seeds/splits.
**P6.3 — Causal tests** — span deletion & feature ablation vs. matched random controls. Verify: top-feature ablation moves s(x) ≥ 3× random-matched control.
**P6.4 — Feature taxonomy + expert study** — classify features (linguistic marker / orthographic / formulaic / NE leakage / genre / source artifact / novel candidate) with Assyriologist. Verify: κ ≥ 0.6 inter-rater on a 50-feature sample.
**P6.5 — Concept-bottleneck dater** — restrict prediction to ~50 named features. Verify: retains ≥ 80% of full-head primary ρ (tests H4).

---

## Phase 7 — Causal objective ladder + generality (Months 14–30)

**P7.1 — Controlled mini-encoder ladder (RQ3)**
- Steps: train 4 matched ~50–100M models on identical cuneiform+multilingual data: NTP / MLM / monolingual denoising seq2seq / translation seq2seq. Same tokenizer, steps, data order.
- Verify: training-token counts equal within 1%; each checkpoint passes the P0.4 pipeline; layerwise ρ scan per objective (the "deepens vs. decays" signature).

**P7.2 — Second-corpus port (RQ5)** — rerun Phases 0–5 on chosen corpus (decision ticket in Y1). Verify: same gates, corpus-appropriate baselines.

**P7.3 — High-resource sanity domain** — dated historical English; expect SSL ≈ linear probe. Verify: registered prediction logged before running.

---

## Milestone / gate summary

```text
G0  (M1):  M.Sc. numbers reproduced exactly            → unlock Phase 1–2
G1  (M2.5): all losses unit-tested, augment engine audited → unlock Phase 3
G2  (M4):  E-MIN — Chrono-Barlow ≥ PLS and ≥2× robustness → unlock Phase 4 (else: diagnose, ablate λs)
G3  (M10): full CJB + semi-supervised ablations frozen   → Paper 1 (Ch. 3)
G4  (M13): calibrated ChronoAtlas demo                   → DH paper / tool release
G5  (M20): SAE feature archaeology + expert study        → Paper 2 (Ch. 4)
G6  (M30): objective ladder + second corpus              → Paper 3 (Ch. 2 & 5), thesis writing
```

## Notes on the Barlow-Twins / JEPA inspiration (for the methods chapter)

- Barlow Twins gives the *loss geometry*: invariance (on-diagonal) + redundancy reduction (off-diagonal), no negatives, no large batches — well suited to 1,202 docs. The PhD's novelty is the reinterpretation **augmentation = confound removal**, turning an anti-shortcut requirement into the SSL objective itself.
- JEPA gives the *prediction target*: latent-space prediction from a degraded view via an EMA target encoder — immune to the token-identity illusion because reconstruction happens in representation space, not token space.
- The ordinal axis borrows from Rank-N-Contrast / RankCORE / soft-rank literature: a differentiable Spearman surrogate aligned with the evaluation metric itself, applied only to relative order from weak labels — never absolute years — with HSIC (CDSSL-style) available as a nonlinear deconfounder.

---

# Addendum (post phase-2, 2026-08) — what the M.Sc. endgame already delivered toward this plan

The plan above was written before the mechanistic follow-up program (F1–F31; see
`HANDOFF_PROMPT.md` on main). Several tickets have since moved from "to build" to
"built or informed":

- **The pairwise data engine exists (→ §2, P2.7, P3.x).** E1 already reframes the
  corpus as ordering: **628,454 ordered fragment pairs** from 1,187 eligible fragments
  (40 rulers), Bradley–Terry on activation differences, both-rulers-held-out folds,
  m=21 quota per ruler pair, ruler-level permutation + dyadic bootstrap inference
  (`v_1/src/phase2/pairs/`). This is exactly the combinatorial supervision §2 promises —
  the CJB rank term can reuse the pair generator and the evaluation harness as-is.
- **The confound menu is now measured, not guessed (→ P1.x).** F28's single-concept
  erasure ladder ranks what actually carries the document "order": ruler identity −.150
  > period −.109 > object type (prism/slab/brick) −.094 > find-spot −.046 > length ≈ 0 —
  and an **untrained twin loses the same at every rung**. Typed ruler-name masks are the
  right first augmentation; object-type deserves a view of its own; length-based views
  matter less than assumed.
- **The token-identity illusion is now mechanistic (→ §2 row 5, RQ4).** The entity year
  axis decomposes into onomastic (name-culture) SAE features, replicated in two
  independent dictionaries and causally verified with rate-matched controls; the ruler
  axis transfers to documents only through ruler identity (E3b) and collapses under
  LEACE. JEPA's "identity carriers are exactly what the augmentations delete" targets
  the mechanism that was actually found, not a hypothesized one.
- **Thesis future-work (vi) is this plan's constructive counterpart.** F23-bridge/F26
  showed no channel carries entity-time features to the document read-out even under
  forcing; training a small adapter to route name-triggered features to the read-out
  position turns that disconnection from finding into design target.
- **Re-pin the P0.4 gate before Phase 0.** The plan's gate uses the 1,202/41-ruler
  frame; the current eligible corpus is 1,187 fragments / 40 rulers / 47 distinct
  years, and the pairwise harness reports macro accuracy with twin/floor context
  (`v_1/src/phase2/pairs/RESULTS.md`) — derive the reproduction numbers from there.
