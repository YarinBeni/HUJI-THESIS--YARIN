# Round 2 — Why did Qwen fail? (Balance → Elicitation → Scale+SAE → Tokenization)

> **See also:** [v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md](v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md) Steps 05–08 (Round 1 findings this plan reacts to) · [v_1/src/sae/plan/PLAN.md](v_1/src/sae/plan/PLAN.md) (Track C SAE plan; Phase 2 dependency) · `papers/txt/Ancient Language papers/Wasserman&Ni, Chronological Attribution and Genre Cohesion Through Computational Lexicometry REVISED.txt` (Phase 0 MC methodology source — Nathan's paper)

**Date:** 2026-05-15 · **Updated:** 2026-05-19 (Phase 0 added per Nathan + Barak meeting)
**Status:** PLANNING — Phases 0, 1a, 1b dispatched in parallel by orchestrator; Phases 2, 3 sequential after Phase 1 verdict
**Round 1 reference:** `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md` Steps 05–08

---

## Research Question
**Is Qwen's failure to linearly encode Akkadian chronology a dataset-imbalance artifact, a representation problem, an elicitation problem, a capacity problem, or a tokenization problem?**

Round 1 headline (ORCC, 893 labeled fragments, 38 rulers, **heavily imbalanced**):
- TF-IDF >> MLM ≈ Random > Qwen
- Qwen 2.5-7B-Instruct fed raw text with no prompt → probe F1 0.117, *worse than random projections*
- Top 3 Sargonid kings = 76.5% of labeled data → Macro-F1 dominated by 3 classes

Round 1 cannot tell us *why* Qwen failed. Round 2 first **removes the imbalance confound** (Phase 0), then eliminates three further plausible confounds — one per phase. Each phase produces thesis-worthy evidence whether it succeeds or fails.

---

## Hypothesis Ladder

| Phase | Hypothesis | What it tests | Cost |
|---|---|---|---|
| **0** | **H0 Imbalance** — Round 1's TF-IDF >> Qwen ranking is a dataset-imbalance artifact; with a balanced subset the picture changes | Wasserman-style MC sampling on top-8 rulers (>15 frags strict) + rerun TF-IDF/MLM/Random/Qwen PLS+CLS | ~1–2 days, one cluster job |
| 1a | **H1a Factual knowledge** — Qwen knows ruler↔period associations as facts | Direct Q&A: "When did ruler X live?" / "Which rulers ruled in period Y?" — no fragment text | ~0.5 day, single cluster job |
| 1b | **H1b Elicitation** — Qwen has chronology knowledge but raw-text framing suppresses it | Prompt variants + re-probe prompted activations on the balanced subset | ~1 day, small cluster |
| 2 | **H2 Capacity/Family** — Qwen 2.5-7B is too small or wrong family; Qwen 3 dense scales out | Qwen 3 at ≥2 sizes (4B, 14B, optional 32B) + official SAE features | 1–2 weeks, cluster-heavy |
| 3 | **H3 Tokenization** — byte-level tokenization fragments Akkadian morphology so the model never has a chance | Akkadian-aware tokenization (existing model OR continue-pretrain w/ extended vocab) | 2–4 weeks |

---

## Where we are
- Round 1 complete: probe pipeline, viz, all infra reusable
- `pls_utils.py`, `05_compute_*` scripts, `06_aggregate_*`, `07_plot_*` are stable
- ORCC labeled set: 893 fragments, 38 rulers, year 7–1132 BCE — single eval surface for all phases
- **Imbalance acknowledged** (Nathan + Barak, 2026-05-19): Phase 0 produces a balanced 8-ruler subset; Phases 1a/1b/2/3 all use that subset as their primary eval surface (full corpus stays as historical baseline)
- Cluster: Schmidt HPC, H100, conda env `thesis`, sbatch workflow

---

## Phase 0 — Dataset Balancing (NEW, prerequisite for Phases 1a/1b/2/3)

**Hypothesis H0:** The TF-IDF >> Qwen ranking from Round 1 is partially an imbalance artifact. With Wasserman-style MC sampling on a balanced ruler subset, methods can be compared fairly. Phase 0 does NOT aim to make Qwen win — it gates whether the *evaluation* is sound before we judge Qwen.

**Source paper for method:** `papers/txt/Ancient Language papers/Wasserman&Ni, Chronological Attribution and Genre Cohesion Through Computational Lexicometry REVISED.txt` (Nathan's own paper — replicate his MC protocol).

### Subset definition (locked 2026-05-19)
Strict `count > 15`. 8 rulers, min class = 21:

| Ruler | Round 1 count | MC class size |
|---|---|---|
| Ashurbanipal | 268 | 21 |
| Sennacherib | 237 | 21 |
| Esarhaddon | 176 | 21 |
| Sargon II | 144 | 21 |
| Nebuchadnezzar II | 87 | 21 |
| Tiglath-pileser III | 75 | 21 |
| Nabonidus | 68 | 21 |
| Sîn-šarru-iškun | 21 | 21 |

Per-draw eval set = 8 × 21 = **168 fragments**. MC draws ≥ 100 (Wasserman value TBD).

### Deliverables
1. **Wasserman MC method note** — `v_1/src/linear_probing/round2_phase0/WASSERMAN_MC.md` summarizing his sampling protocol (number of draws, what's aggregated, error bars).
2. **Balanced subset materializer** — `v_1/src/linear_probing/round2_phase0/build_balanced_subset.py`. Takes `orcc_corpus.parquet`, returns N MC draws as fragment-ID lists with deterministic seed.
3. **Rerun probes on each MC draw**: TF-IDF (PLS+CLS, char n-grams + ruler), MLM (PLS+CLS), Random Qwen, pretrained Qwen — same drivers as Round 1 (`05_compute_*`), new method tag `mc{draw_id}`.
4. **Aggregate** across MC draws: mean ± std Macro-F1 per method, layer, target. Mirror Round 1 output schema so `07_plot_*` works.
5. **Compare to Wasserman's paper TF-IDF result** as the validation anchor.

### Success criterion (pre-committed) — Phase 0 Gate
**TF-IDF Macro-F1 on the balanced subset matches or exceeds the level reported in Wasserman's paper on a comparable subset** (exact threshold to be set after reading the paper; placeholder ≥ 0.50 Macro-F1).

> **Why this gate**: if TF-IDF still gets the strong results Wasserman reported, the balancing is sound and we have a *trustworthy evaluation surface* to judge Qwen on. If TF-IDF tanks too, the issue is our pipeline, not the model — and we fix that first.

### Decision gate
- **Pass →** Balanced subset is the validated eval surface. Phase 1a, 1b (and downstream Phase 2/3) use it.
- **Fail →** Stop. Diagnose pipeline/balancing before any other phase runs.

### Phase 0 runtime + parallel notes
- No new prompts → no prompt-approval gate.
- Single cluster job (or two: one for activation reuse from Round 1, one for fresh TF-IDF runs which need no GPU).
- Can run in parallel with Phase 1a/1b drafting (prompt-drafter does NOT need Phase 0 output).

---

## Phase 1a — Factual Knowledge Probe (NEW, ~0.5 day)

**Hypothesis H1a:** Qwen has the basic factual knowledge that ruler X reigned in period Y. This is a *sanity check* — if Qwen can't answer "when did Ashurbanipal live?", H1 (latent-but-inaccessible) is dead on arrival.

### Deliverables
1. **Prompt file** `v_1/src/linear_probing/round2_phase1a/prompts/{kp0,kp1,kp2}.md`:
   - kp0: "When did the Akkadian/Assyrian/Babylonian ruler X live? Answer with a year range BCE."
   - kp1: "Which rulers reigned during period P?" (period ∈ {OB, NA, LB})
   - kp2: control — a fake ruler name, to detect hallucination rate
2. **Direct Q&A inference** on all 8 Phase 0 rulers × prompt variants. No fragment text — pure factual recall.
3. **Score**: extract year range, compare to ground-truth reign dates (compile from Wikipedia / Round 1's `ruler_year` mapping).

### Success criterion (pre-committed)
**Qwen returns a date overlapping the true reign for ≥ 6 of 8 rulers** (kp0). Hallucination rate on kp2 < 30%.

### Decision gate
- **Pass →** Qwen has the facts. H1b's "latent-but-inaccessible" remains viable. Phase 1b is the right next test.
- **Fail →** Qwen doesn't know the rulers as facts. H1 is implausible; Phase 1b may still be useful as a control but the expected outcome is "no improvement." Phase 2 (scale) becomes the priority.

### Phase 1a notes
- **PROMPT APPROVAL GATE** — Yarin must approve `kp0/kp1/kp2` before any sbatch job submits.
- Can run *fully in parallel* with Phase 0 (no dependency on balanced subset — uses fixed ruler list).
- Cheap: 8 rulers × 3 prompts × N samples ≈ tens of generations.

---

## Phase 1b — Elicitation on Balanced Subset (rewritten, ~1 day)

**Hypothesis H1b:** Task framing suppresses Qwen's latent knowledge. Test on the **Phase 0 balanced subset** so we're not chasing imbalance artifacts.

**Subagent:** `phase1-eliciter` — see `.claude/agents/phase1-eliciter.md`.

### Deliverables
1. **Direct-fragment-answer baseline.** For each fragment in the balanced subset, ask Qwen 2.5-7B-Instruct (zero-shot, no prompt) "What year (BCE) was this Akkadian inscription written?" and "Which ruler authored it?" Parse + score. *(Distinct from Phase 1a, which asks about rulers *without* any fragment text.)*
2. **Prompt variants** (each run as a separate sweep, not combined):
    - a. Zero-shot with explicit task framing ("This is an Akkadian royal inscription. Estimate the year BCE.")
    - b. Few-shot (k=5) with labeled in-context examples (held out from eval set)
    - c. Chain-of-thought prefix ("Reason about palaeography, ruler names, and titulary, then answer.")
3. **Probe-on-prompted activations.** Extract CLS at the last token of the *fragment span* (not prompt boilerplate). Re-run PLS-DA + LogReg probes on the balanced subset with MC sampling per Phase 0.
4. **The 2x2 disentangler table** (combined with Phase 1a's knowledge probe row):

    | 1a Knows facts | 1b Direct | 1b Probe | Interpretation |
    |---|---|---|---|
    | ✓ | ✓ | ✓ | Encodes + verbalizes (verify no contamination) |
    | ✓ | ✗ | ✓ | Latent but inaccessible → prompting wins |
    | ✓ | ✓ | ✗ | Non-linear access → flag, try MLP probe |
    | ✓ | ✗ | ✗ | Knows facts but can't apply to text → proceed to Phase 2 |
    | ✗ | * | * | Doesn't know rulers at all → H1 dead → Phase 2 |

### Success criterion (pre-committed)
**Direct-answer Macro-F1 ≥ 0.25 (ruler) on balanced subset, OR probe-on-prompted Macro-F1 ≥ Qwen Phase-0-balanced baseline + 0.05** (relative to balanced baseline, not Round 1's imbalanced 0.117).

### Decision gate
- **Pass →** Negative result was an elicitation artifact. Write up: "Qwen *does* encode Akkadian chronology when prompted." Phase 2 becomes optional/confirmatory.
- **Fail →** H1b ruled out. The knowledge isn't there to elicit. **Go to Phase 2.**

### Runtime + approval rules
- **Inference runtime:** cluster Qwen 2.5-7B-Instruct (already loaded for Round 1 embedding extraction). No vLLM-local, no OpenRouter. Reuse `v_1/src/embeddings/` model-loading code; add a generation path on top of it.
- **PROMPT APPROVAL GATE — MANDATORY.** Before any inference job runs, every prompt template (system+user, plus parse instructions) must be written into `v_1/src/linear_probing/round2_phase1b/prompts/{pv0,pv1,pv2,pv3}.md` and explicitly approved by Yarin. The agent does NOT submit sbatch jobs until approval is logged in `prompts/APPROVED.md`. No exceptions.
- **CLS pooling** convention as Round 1 (last token of *fragment span*, not prompt boilerplate). Track token offsets carefully.
- **Layer sweep:** L0, L15 (Round 1 best), L-1 only. Don't sweep all 28 layers.
- **Eval set:** Phase 0 balanced subset (MC draws). Output schema must match Round 1 so `06_aggregate_*` and `07_plot_*` work unchanged.
- **Dependency:** activation extraction depends on prompts being approved, but does NOT depend on Phase 0 completion — extract on the full corpus, then mask to balanced subset at probing time. This is what lets Phases 0/1a/1b run in parallel.

---

## Phase 2 — Scale + SAE (expensive, 1–2 weeks)

**Hypothesis H2:** Capacity/family. Qwen 3 dense models > Qwen 2.5-7B; effect scales with size.

**Subagent:** `phase2-scaler-sae` — see `.claude/agents/phase2-scaler-sae.md`.

### Deliverables
1. **Extract residual stream** at all layers for Qwen 3 dense at ≥ 2 sizes:
    - 3-4B dense
    - 3-14B dense
    - (optional) 3-32B dense — only if 14B shows positive scaling
    - **Skip MoE variants** — confounds the scaling axis
2. **Re-run probing pipeline** (PLS-DA ruler, PLS year regression, LogReg CLS) — reuses `05_compute_*` with new method tag.
3. **Scaling plot:** probe F1 vs parameter count. Pre-register: positive monotonic slope is the win.
4. **SAE analysis** using Qwen 3's official SAE (verify availability; fallback: train a small dictionary on ORCC residuals):
    - Sparse LogReg on SAE features → how many features explain ≥ 80% of probe F1?
    - Inspect top features: temporal? onomastic? geographic? Use differential bigram analysis on max-activating fragments.
    - Project Round 1's PLS-Ruler direction onto SAE decoder columns → align mechanistic features to behavioral probe.

### Success criterion (pre-committed)
**Largest model probe Macro-F1 ≥ 0.30 AND positive scaling slope across ≥ 2 sizes AND SAE finds ≤ 50 features explaining ≥ 80% of probe F1**

### Decision gate
- **Pass →** "Temporal representation in Akkadian emerges with scale in dense Qwen 3 and is sparse + interpretable via SAE." Tracks B+C unify. This becomes the thesis headline.
- **Fail (flat scaling) →** Capacity not the issue. **Go to Phase 3.**

### Risks / gotchas
- Qwen 3 SAE may be layer-restricted (Arditi-style — only specific layers). Check HF before designing the layer sweep.
- 14B+ extraction needs careful batching for H100 80GB. Reuse `v_1/src/embeddings/` infrastructure.
- SAE feature interpretation is the slowest part — budget 3-4 days for it alone.

---

## Phase 3 — Tokenization / Finetune (most expensive, 2–4 weeks)

**Hypothesis H3:** Byte-level tokenization fragments Akkadian morphology; signal is lost at input.

**Subagent:** `phase3-tokenizer` — see `.claude/agents/phase3-tokenizer.md`.

### Deliverables (cheapest first)
1. **Survey existing Akkadian-aware models** on HuggingFace (Gutherz, Wisnom, ORACC-trained checkpoints, BabyLM-Akkadian if it exists). Run our probing pipeline on whatever is available.
2. **If none usable: continue-pretrain Qwen** with extended vocab:
    - Train SentencePiece tokenizer on full Akkadian corpus (SEAL + DLL + LBPL + ORCC + ORACC text)
    - Extend Qwen 3 embedding matrix; resize LM head
    - Continue-pretrain on Akkadian (MLM or causal LM) for 1–3 epochs
    - Re-run probe
3. **Train from scratch** — only if (1) and (2) both fail. Small model (~100M-400M params), Akkadian-only.

### Success criterion (pre-committed)
**Akkadian-aware model probe Macro-F1 ≥ 0.30 AND gap to byte-level Qwen ≥ 0.10 (controlled: same arch, same data, only vocab differs)**

### Decision gate
- **Pass →** Tokenization was the bottleneck. Strong digital-humanities methodological contribution.
- **Fail →** All three confounds ruled out. The negative result is robust: **Akkadian chronology is not linearly recoverable from frontier LLM representations, regardless of prompting, scale, or tokenization.** Publishable as a boundary finding for LLM interpretability on ancient low-resource languages.

---

## Decision Tree

```
┌─ Phase 0 (~1-2 day) ── balance ORCC, MC-resample, re-run probes ─── TF-IDF matches Wasserman's paper?
│     │ fail → STOP, fix pipeline/balancing
│     │ pass ↓ (balanced subset now the validated eval surface)
│
├─ Phase 1a (~0.5 day, PARALLEL w/ 0+1b) ── Qwen knows ruler↔period facts? (no fragment text)
│     │ fail → H1 dead, prioritize Phase 2
│     │ pass ↓ (latent-but-inaccessible remains viable)
│
└─ Phase 1b (~1 day, PARALLEL w/ 0+1a until probing) ── elicitation lifts Qwen on balanced subset?
      │ pass → write up "Qwen encodes Akkadian chronology when prompted"; Phase 2 optional
      │ fail ↓
Phase 2 (1-2 wk) ─── scale + SAE on Qwen 3?
   │ pass → unified Track B+C thesis ★
   │ fail ↓
Phase 3 (2-4 wk) ─── tokenization?
   │ pass → DH-flavored contribution
   │ fail → robust negative result, still thesis-worthy
```

**Phases 0, 1a, 1b are dispatched simultaneously by the orchestrator.** Phase 1b activation extraction starts as soon as Phase 1b prompts are approved (independent of Phase 0); Phase 1b *probing* (on balanced subset) blocks on Phase 0 subset materialization. Phase 2 starts only after Phase 1 verdict.

---

## Agent Execution Model — role-based, parallel within a phase

The main Claude session is the orchestrator. Within each phase, work decomposes into parallel streams; each stream is dispatched to a **role-based worker agent** via the `Agent` tool. Workers live in `.claude/agents/` and are reusable across phases.

### Role-based workers
| Role | Agent file | What it does |
|---|---|---|
| Prompt design | `.claude/agents/prompt-drafter.md` | Drafts prompt variants; runs the **user-approval gate** before any inference fires |
| Eval code | `.claude/agents/eval-harness-builder.md` | Builds `run_generation.py` + `parse_predictions.py` + `score.py` on top of `v_1/src/embeddings/` model loader |
| Cluster jobs | `.claude/agents/cluster-job-runner.md` | Writes sbatch scripts + prints the exact `sbatch ...` commands for Yarin to run; never ssh's |
| Synthesis | `.claude/agents/phase-synthesizer.md` | Pulls worker outputs into a phase REPORT.md ending in `## Verdict: PASS/FAIL` against the pre-committed criterion |

### Combined Phase 0 + 1a + 1b dispatch waves (model: opus 4.7 orchestrator, sonnet 4.6 workers)

The orchestrator (this assistant, Opus 4.7) dispatches all workers below as Sonnet 4.6 sub-agents in parallel where possible. Yarin gates approvals and runs cluster jobs himself.

**Wave 1 — research + drafting (all parallel; no dependencies on each other):**
| ID | Agent | Stream | Output |
|---|---|---|---|
| W1.A | general-purpose | Read Wasserman paper, extract MC method (draws count, aggregation, error reporting) | `round2_phase0/WASSERMAN_MC.md` |
| W1.B | general-purpose | Compile ruler→reign-year mapping for 8 Phase-0 rulers + Round 1 ruler→year dict reuse | `round2_phase1a/ruler_reigns.json` |
| W1.C | prompt-drafter | Phase 1a knowledge-probe prompts (kp0/kp1/kp2) + parse spec | `round2_phase1a/prompts/{kp0,kp1,kp2}.md` |
| W1.D | prompt-drafter | Phase 1b fragment-classification prompts (pv0–pv3: direct, zero-shot, few-shot, CoT) + parse spec | `round2_phase1b/prompts/{pv0..pv3}.md` |

**SYNC POINT** — Yarin reviews W1.A summary; **approves both prompt sets** (W1.C, W1.D) into `prompts/APPROVED.md`; confirms Phase 0 subset locked.

**Wave 2 — implementation + sbatch drafting (all parallel after Wave 1):**
| ID | Agent | Stream | Output |
|---|---|---|---|
| W2.A | general-purpose | Implement `build_balanced_subset.py` per W1.A's MC spec | `round2_phase0/build_balanced_subset.py` |
| W2.B | eval-harness-builder | Phase 1a generation+parse+score harness | `round2_phase1a/run_kp.py`, `parse_kp.py`, `score_kp.py` |
| W2.C | eval-harness-builder | Phase 1b generation+parse harness (reuses 1a infra) + activation-extraction hook | `round2_phase1b/run_pv.py`, `extract_prompted_acts.py` |
| W2.D | cluster-job-runner | Three sbatch scripts: P0 probes, P1a inference, P1b inference+activations | `round2_phase{0,1a,1b}/sbatch/*.sbatch` + commands |

**SYNC POINT** — Yarin submits 3 sbatch jobs, reports job IDs.

**Wave 3 — post-cluster analysis (parallel per phase, kick off as each job completes):**
| ID | Trigger | Agent | Stream |
|---|---|---|---|
| W3.A | P0 done | general-purpose | Aggregate TF-IDF/MLM/Random/Qwen across MC draws → mean±std tables, plots |
| W3.B | P1a done | general-purpose | Score knowledge-probe outputs against reign-year mapping; compute hallucination rate on kp2 |
| W3.C | P1b done | general-purpose | Probe prompted activations on Phase 0 balanced subset (needs W3.A's subset draws) |

**Wave 4 — synthesis:**
| ID | Agent | Stream |
|---|---|---|
| W4 | phase-synthesizer | Combined `REPORT.md` with PASS/FAIL on Phase 0, 1a, 1b gates + the 2x2 disentangler table |

Output contract: every worker writes to `v_1/src/linear_probing/results/orcc_round2_phase{0,1a,1b}/` and returns a ≤ 250-word summary to the orchestrator.

### Critical-path estimate
1. Wave 1 (~1 hr in parallel) → Yarin approval (~15 min) → Wave 2 (~2 hr in parallel) → Yarin cluster submission → **cluster runtime is the long pole** (P0 + P1b each multi-hour; P1a small) → Wave 3 (~1 hr) → Wave 4 (~30 min).
2. Wall-clock from green-light to verdict: ~1 day of human attention + overnight cluster.

---

## Output Layout
```
v_1/src/linear_probing/results/
├── orcc_round2_phase0/
│   ├── WASSERMAN_MC.md           # method extraction from Nathan's paper
│   ├── balanced_subset/          # MC draw fragment-ID lists (seeded)
│   ├── probes/                   # TF-IDF/MLM/Random/Qwen on each draw
│   ├── figures/                  # mean±std plots
│   └── REPORT.md                 # Phase 0 verdict
├── orcc_round2_phase1a/
│   ├── prompts/{kp0,kp1,kp2}.md + APPROVED.md
│   ├── ruler_reigns.json         # ground truth
│   ├── direct_kp/                # generation outputs
│   ├── scores/
│   └── REPORT.md
├── orcc_round2_phase1b/
│   ├── prompts/{pv0..pv3}.md + APPROVED.md
│   ├── direct_answers/           # model JSON outputs per fragment
│   ├── prompted_activations/     # CLS at fragment last-token under each prompt variant
│   ├── probes/                   # probes on Phase 0 balanced subset
│   ├── figures/
│   └── REPORT.md
└── orcc_round2_phase{2,3}/       # post-Phase-1-verdict
    ├── probes/
    ├── sae/                      # phase 2 only
    ├── figures/
    └── REPORT.md
```

---

## Open Questions to Resolve Before Wave 1 Dispatch
1. ~~Inference runtime~~ — RESOLVED 2026-05-15: cluster Qwen 2.5-7B reuses `v_1/src/embeddings/` model-loading; generation path added on top.
2. ~~Phase 0 ruler subset~~ — RESOLVED 2026-05-19: strict `>15` → 8 rulers, min class 21.
3. ~~Phase 0 balancing method default~~ — RESOLVED 2026-05-19: follow Wasserman MC exactly (W1.A reads paper first).
4. ~~Phase 0/1a/1b parallelism~~ — RESOLVED 2026-05-19: balanced subset for all probes; full corpus stays as historical baseline only.
5. Phase 0 success threshold — placeholder ≥ 0.50 Macro-F1 for TF-IDF; **final number set after W1.A reports Wasserman's number**.
6. Contamination check — deferred; revisit only if Phase 1a/1b direct-answer F1 is suspiciously high.
7. Prompts (Phase 1a *and* Phase 1b) must be drafted + presented for approval before any cluster job submits.
