# Round 2 — Why did Qwen fail? (Elicitation → Scale+SAE → Tokenization)

> **See also:** [v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md](v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md) Steps 05–08 (Round 1 findings this plan reacts to) · [v_1/src/sae/plan/PLAN.md](v_1/src/sae/plan/PLAN.md) (Track C SAE plan; Phase 2 dependency)

**Date:** 2026-05-15
**Status:** PLANNING — phases run sequentially, gated by success criteria
**Round 1 reference:** `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md` Steps 05–08

---

## Research Question
**Is Qwen's failure to linearly encode Akkadian chronology a representation problem, an elicitation problem, a capacity problem, or a tokenization problem?**

Round 1 headline (ORCC, 893 labeled fragments, 38 rulers):
- TF-IDF >> MLM ≈ Random > Qwen
- Qwen 2.5-7B-Instruct fed raw text with no prompt → probe F1 0.117, *worse than random projections*

Round 1 cannot tell us *why* Qwen failed. Round 2 eliminates three plausible confounds — one per phase. Each phase rules out a hypothesis and produces thesis-worthy evidence whether it succeeds or fails.

---

## Hypothesis Ladder

| Phase | Hypothesis | What it tests | Cost |
|---|---|---|---|
| 1 | **H1 Elicitation** — Qwen has the knowledge but raw-text framing suppresses it | Prompt variants + re-probe prompted activations + direct-answer baseline | ~1 day, local + small cluster |
| 2 | **H2 Capacity/Family** — Qwen 2.5-7B is too small or wrong family; Qwen 3 dense scales out | Qwen 3 at ≥2 sizes (4B, 14B, optional 32B) + official SAE features | 1–2 weeks, cluster-heavy |
| 3 | **H3 Tokenization** — byte-level tokenization fragments Akkadian morphology so the model never has a chance | Akkadian-aware tokenization (existing model OR continue-pretrain w/ extended vocab) | 2–4 weeks |

---

## Where we are
- Round 1 complete: probe pipeline, viz, all infra reusable
- `pls_utils.py`, `05_compute_*` scripts, `06_aggregate_*`, `07_plot_*` are stable
- ORCC labeled set: 893 fragments, 38 rulers, year 7–1132 BCE — single eval surface for all phases
- Cluster: Schmidt HPC, H100, conda env `thesis`, sbatch workflow

---

## Phase 1 — Elicitation (cheap, ~1 day)

**Hypothesis H1:** Task framing suppresses Qwen's latent knowledge.

**Subagent:** `phase1-eliciter` — see `.claude/agents/phase1-eliciter.md`.

### Deliverables
1. **Direct-answer baseline.** For each labeled fragment, ask Qwen 2.5-7B-Instruct (zero-shot, no prompt) "What year (BCE) was this Akkadian inscription written?" and "Which ruler authored it?" Parse + score.
2. **Prompt variants** (each run as a separate sweep, not combined):
    - a. Zero-shot with explicit task framing ("This is an Akkadian royal inscription. Estimate the year BCE.")
    - b. Few-shot (k=5) with labeled in-context examples (held out from eval set)
    - c. Chain-of-thought prefix ("Reason about palaeography, ruler names, and titulary, then answer.")
3. **Probe-on-prompted activations.** Extract CLS at the last token of the *fragment span* (not prompt boilerplate). Re-run PLS-DA + LogReg probes.
4. **The 2x2 disentangler table:**

    | Model direct | Probe | Interpretation |
    |---|---|---|
    | ✓ | ✓ | Encodes + verbalizes (need to verify it's not data contamination) |
    | ✗ | ✓ | Latent but inaccessible → prompting wins |
    | ✓ | ✗ | Non-linear access → flag, try MLP probe |
    | ✗ | ✗ | Doesn't know → proceed to Phase 2 |

### Success criterion (pre-committed)
**Direct-answer Macro-F1 ≥ 0.25 (ruler) OR probe-on-prompted Macro-F1 ≥ Qwen Round 1 + 0.05**

### Decision gate
- **Pass →** Negative result was an elicitation artifact. Write up: "Qwen *does* encode Akkadian chronology when prompted." Phase 2 becomes optional/confirmatory.
- **Fail →** H1 ruled out. The knowledge isn't there to elicit. **Go to Phase 2.**

### Runtime + approval rules
- **Inference runtime:** cluster Qwen 2.5-7B-Instruct (already loaded for Round 1 embedding extraction). No vLLM-local, no OpenRouter. Reuse `v_1/src/embeddings/` model-loading code; add a generation path on top of it.
- **PROMPT APPROVAL GATE — MANDATORY.** Before any inference job runs, every prompt template (system+user, plus parse instructions) must be written into `v_1/src/linear_probing/round2_phase1/prompts/{pv0,pv1,pv2,pv3}.md` and explicitly approved by Yarin. The agent does NOT submit sbatch jobs until approval is logged in `prompts/APPROVED.md`. No exceptions.
- **CLS pooling** convention as Round 1 (last token of *fragment span*, not prompt boilerplate). Track token offsets carefully.
- **Layer sweep:** L0, L15 (Round 1 best), L-1 only. Don't sweep all 28 layers.
- **Output schema** must match Round 1 so `06_aggregate_*` and `07_plot_*` work unchanged.

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
Phase 1 (~1 day) ─── elicitation?
   │ pass → write up, phase 2 optional
   │ fail ↓
Phase 2 (1-2 wk) ─── scale + SAE?
   │ pass → unified Track B+C thesis ★
   │ fail ↓
Phase 3 (2-4 wk) ─── tokenization?
   │ pass → DH-flavored contribution
   │ fail → robust negative result, still thesis-worthy
```

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

### Phase 1 dispatch waves
- **Wave 1 (parallel):** `prompt-drafter` (B) ∥ `eval-harness-builder` (A). Wave 1 gates the rest — B has the user approval gate, A produces the sanity check.
- **Wave 2 (parallel, after Wave 1):** `cluster-job-runner` × 2 — one job for direct-answer inference (C), one for prompted-activation extraction (D). Yarin runs the sbatch commands, reports back job IDs.
- **Wave 3 (parallel, after cluster jobs return):** `eval-harness-builder` re-invoked to score direct answers (E) ∥ Round 1 `05_compute_*` scripts to re-probe (F).
- **Wave 4:** `phase-synthesizer` writes `REPORT.md` (G).

Output contract: every worker writes to `v_1/src/linear_probing/results/orcc_round2_phase1/` and returns a ≤ 250-word summary to the orchestrator.

---

## Output Layout
```
v_1/src/linear_probing/results/orcc_round2_phase{1,2,3}/
├── direct_answers/      # phase 1 only: model JSON outputs per fragment
├── probes/              # PLS + CLS results (same schema as Round 1)
├── sae/                 # phase 2 only: feature analysis
├── figures/             # plots, scaling curves
└── REPORT.md            # phase verdict: pass/fail vs success criterion
```

---

## Open Questions to Resolve Before Phase 1 Starts
1. ~~Inference runtime~~ — RESOLVED 2026-05-15: cluster Qwen 2.5-7B reuses `v_1/src/embeddings/` model-loading; generation path added on top.
2. Contamination check — deferred; revisit only if direct-answer F1 is suspiciously high.
3. Prompts must be drafted + presented for approval before any cluster job submits.
