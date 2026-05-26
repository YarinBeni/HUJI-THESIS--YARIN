# Round 3 — Geodesic / Manifold Readout of Akkadian Temporal Structure

> **See also:** [PLAN_round2_qwen_diagnosis.md](PLAN_round2_qwen_diagnosis.md) (Round 2, immediate predecessor) · [PLAN_viz_extension.md](PLAN_viz_extension.md) (UMAP viz plan — superseded by Phase B below) · [v_1/src/linear_probing/results/orcc_round2_REPORT.md](v_1/src/linear_probing/results/orcc_round2_REPORT.md) (Round 2 leaderboard incl. Thalesian year R²) · [v_1/src/sae/plan/PLAN.md](v_1/src/sae/plan/PLAN.md) (Track C SAE — sequenced *after* this round)

**Date:** 2026-05-23
**Status:** PLANNING — start after Round 2 closes (job 8477 aggregated)
**Advisor input:** external neural-geometry reviewer (Goodfire manifold-steering + Origins-of-Representation-Manifolds framing)

---

## Motivation

Round 1 and Round 2 measured every method with a **linear** probe — ridge classification (CLS, Macro-F1) and PLS regression (PLS, Spearman/R²/MAE). The strongest result we found is that **Thalesian cuneiBase-400m at L12 tier0/mean is the only model with positive year R² (+0.105)**, while delivering Spearman 0.467 and MAE 75 yr on lower-bound year labels.

That pattern — high Spearman but modest R² — is exactly what a curved-but-real temporal manifold would produce under a linear probe. A linear probe asks *"is there one global straight direction that reads time?"* and necessarily flattens any curve. The neural-geometry reviewer proposes weaker, more honest readouts:

> *Manifold readout asks: even if time bends through activation space, do nearby activations form an ordered path?*

If this hypothesis holds, the headline of Round 3 is:

**"Linear probing systematically understated the temporal structure in Akkadian-finetuned representations; a manifold readout recovers a clean 1D historical timeline that linear regression cannot see."**

That is a much stronger thesis claim than Round 2's "Thalesian beats Qwen on year regression."

---

## Research Question

**Does an Akkadian-finetuned encoder's hidden state contain a 1D temporal manifold along which texts can be ordered by lower-bound year, and is that ordering preserved after controlling for ruler/archive/genre shortcuts?**

Sub-questions:
- **Q1 (existence):** Per (method, cleaning, pool, layer), does a kNN-graph geodesic 1D coordinate produce a higher pairwise-order accuracy and Spearman against year than the corresponding linear ridge probe?
- **Q2 (layer choice bias):** Does the layer selected by linear probe match the layer selected by geodesic readout? If not, by how much, and in which direction?
- **Q3 (honesty):** Does the manifold survive leave-one-ruler-out and leave-one-archive-out evaluation, or is it a ruler/archive cluster trajectory in disguise?
- **Q4 (cross-method):** Does the geodesic gap between Thalesian and {Qwen pretrained, MLM, Random-Qwen, TF-IDF} grow, shrink, or invert relative to the linear gap?

---

## Pre-committed predictions (lock before running)

To avoid post-hoc storytelling, here is what each result would *mean* — committed in writing now:

| Outcome | Interpretation |
|---|---|
| Geodesic Spearman > linear PLS Spearman by ≥0.10 on Thalesian cuneiBase L12 | Linear probe understated temporal signal; thesis headline holds |
| Geodesic ≈ linear PLS within ±0.05 | Signal is approximately linear at that layer; manifold is a visualization not a discovery |
| Geodesic < linear PLS | PLS exploited supervised projection in a way unsupervised geometry cannot; reframe as "PLS finds the signal even when geometry is messy" |
| Best geodesic layer ≠ best linear PLS layer for ≥2 of 5 methods | Layer-selection bias in Round 1/2 is real; document and add geodesic-best-layer column to all leaderboards |
| Leave-ruler-out drop > 0.20 in pairwise accuracy | Manifold is partially a ruler-cluster trajectory; report as confound, not refutation |
| Leave-ruler-out drop < 0.10 | Strong evidence that the manifold is genuinely temporal beyond ruler-entity shortcuts |
| Pairwise-order accuracy >0.75 on Thalesian for pairs separated by >100 yr | Operationally useful dating tool — promote to thesis demo |
| Pairwise-order accuracy <0.60 on every method | Honest null result; either ORCC labels too noisy or 1D manifold doesn't exist at this scale |

---

## What we already have (no new compute needed for Phases A–C)

| Input | Path | Status |
|---|---|---|
| Per-layer activations, all methods | `v_1/src/linear_probing/results/orcc__embed/activations/<method>_<cleaning>_<pool>/layer_NN.npz` | ✅ for Qwen pretrained, MLM Aeneas, Thalesian akk300m, Thalesian cunei400m × {tier0, maximal} × {mean, last} × all layers. **Random-Qwen has a known activation-path gap — see Data Inventory section below.** |
| Year labels (lower bound) | `v_1/data/evaluation/corpora/orcc_corpus.parquet` col `year` | ✅ 1,193 / 1,202 fragments labeled |
| Ruler labels (for honesty pass) | same parquet, col `ruler` | ✅ |
| Linear PLS baseline numbers per layer | `v_1/src/linear_probing/results/orcc__probe_pls/pls_results_*.json` | ✅ for free side-by-side comparison |
| Linear best-layer per method | `v_1/src/linear_probing/results/orcc__probe_pls/pls_best_layers.json` | ✅ — to be augmented, not replaced, with `geodesic_best_layers.json` |
| Round 2 unified report | `v_1/src/linear_probing/results/orcc_round2_REPORT.md` | ✅ — Round 3 appends a section, doesn't rewrite |

The only thing that would require cluster work is the **masked-text robustness pass** (advisor's Case A, Phase C step 2 below): re-extracting activations with explicit ruler-name / date-formula masking applied to the input text. Tier0 cleaning already strips a lot, so the marginal value of an explicit mask is uncertain — defer until Phase C dev numbers say whether it's worth a job.

---

## Data Inventory — what's covered, what's missing

Audited 2026-05-23 across both regimes (CLS = logistic regression on hidden states → ruler classification; PLS = year-PLS-DA → ruler Macro-F1). Coverage is excellent on the imbalanced surface, sparse on the balanced-MC surface, and Random-Qwen has a path-forensics issue. All three matter for Round 3.

### R1 imbalanced coverage: 38 / 38 cells (100%)

Every method × cleaning × pool × regime combination has an R1 number. MLM Aeneas is the only "missing" cell and it's by design (Aeneas tokenizer is sign-level, so cleaning ≠ tier0 and pool ≠ mean are not defined for it). Headline numbers carried into Round 3 as the linear-probe baseline column:

- **CLS:** TF-IDF tier0 R1 F1 0.3262 · Thalesian cuneiBase maximal/mean L12 R1 F1 0.2625 · Thalesian cuneiBase tier0/mean L12 R1 F1 0.2103 · Random-Qwen tier0/mean L1 0.2350 · MLM Aeneas tier0/mean L0 0.2195 · Qwen-7B tier0/mean L0 0.1167.
- **PLS:** TF-IDF tier0 R1 F1 0.1128 · Random-Qwen tier0/mean L0 0.1147 · Thalesian cuneiBase tier0/mean L12 0.1143 · Qwen-7B tier0/mean L0 0.1113 · MLM Aeneas tier0/mean L16 0.1064 · Thalesian AKK_300m tier0/mean L4 0.1081.

### Balanced MC coverage: 12 / 44 cells (27%) — sparse by design

`run_mc_probes.py` hardcodes `cleaning="tier0"` and `pooling="mean"` for all activation methods, so balanced MC only runs on the canonical regime. Maximal-mean, tier0-last, maximal-last were never planned for MC. The 13 missing (method, cleaning, pool) combos × 2 regimes = ~26 uncovered cells.

| Method | Cleaning | Pool | Missing |
|---|---|---|---|
| Random-Qwen | tier0 | mean | CLS-MC, PLS-MC |
| Random-Qwen | tier0 | last | CLS-MC, PLS-MC |
| Random-Qwen | maximal | mean | CLS-MC, PLS-MC |
| Random-Qwen | maximal | last | CLS-MC, PLS-MC |
| Qwen-7B | tier0 | last | CLS-MC, PLS-MC |
| Qwen-7B | maximal | mean | CLS-MC, PLS-MC |
| Qwen-7B | maximal | last | CLS-MC, PLS-MC |
| Thalesian AKK_300m | tier0 | last | CLS-MC, PLS-MC |
| Thalesian AKK_300m | maximal | mean | CLS-MC, PLS-MC |
| Thalesian AKK_300m | maximal | last | CLS-MC, PLS-MC |
| Thalesian cuneiBase-400m | tier0 | last | CLS-MC, PLS-MC |
| Thalesian cuneiBase-400m | maximal | mean | CLS-MC, PLS-MC |
| Thalesian cuneiBase-400m | maximal | last | CLS-MC, PLS-MC |

**Decision for Round 3:** do NOT backfill these MC cells unconditionally. Round 3's primary metrics are geodesic (pairwise-order accuracy, neighbor purity, centroid order), which are computed locally on the activations directly — they do not need MC bootstraps to be reportable. Reserve a balanced-MC backfill for *only* the (method, cleaning, pool) combo that wins Phase B, so the leaderboard's headline row has both a linear-MC and a geodesic-MC number. That's at most 2 new MC sbatch jobs (one CLS, one PLS) instead of 26.

### Random-Qwen P0 data gap — forensics + three fix options

Cluster job 8198 wrote 400 `random_{cls,pls}__mc_balanced__draw{000..199}.json` files, each with `results: {}` (empty). The draw metadata is populated (168 fragment IDs per draw); the probes themselves never ran. Root cause: `run_mc_probes.py:_load_orcc_activations("random", layer, acts_base)` returned `None` for every layer because the random activations weren't at the path it looked.

**Path searched:** `v_1/src/linear_probing/results/orcc_round1/activations/random_tier0_mean/layer_NN.npz`

**Three candidate paths where they might actually be** (Yarin verifies on cluster):

```bash
ls v_1/src/linear_probing/results/orcc_round1/activations/random_tier0_mean/ \
   v_1/src/linear_probing/results/orcc__embed/activations/random_tier0_mean/ \
   v_1/src/linear_probing/results/activations/qwen2.5-7b-instruct-random/tier0/ 2>&1
```

**Fix options** (cheapest first):

1. **Symlink fix (~2 min):** if path 2 or 3 has the files, symlink to the expected path. The dual-path fallback added during Phase 3 extension already tries `orcc__embed/activations/random_tier0_mean/` automatically, so re-running `run_mc_probes.py --probes random_pls,random_cls` may just work without a symlink.
2. **Re-extract (~30 min cluster):** if no path has the files, re-run `01b_extract_random_baseline.py` on 1,202 fragments × 29 layers (single H100). Then submit the MC probes.
3. **Drop Random-Qwen from MC (0 min):** if Round 3 Phase B has the geodesic readout on the imbalanced surface working for Random-Qwen (which only needs the activations, not the MC sweep), the balanced-MC gap becomes a footnote rather than a blocker. This is the **default plan** — fix only if Phase B unexpectedly needs it.

Random-Qwen is also the trickiest method to interpret in this round: it's an initialization control, not a real signal source. If its geodesic readout outperforms Qwen pretrained (as it does on imbalanced R1 — 0.2350 vs 0.1167 CLS F1), that's already a thesis-worthy finding about how much "useful structure" raw initialization produces — and it doesn't depend on balanced MC.

---

## Cluster parallelism policy

When Round 3 work needs cluster GPUs (Phase C masked extraction, Phase E SAE / Qwen 3 scale sweeps), **fan out over many small parallel sbatch jobs**, not one sequential job. Cluster has 64 H100s and 832 CPUs; the documented best practice (`v_1/src/cluster/README.md` line 188) is *"With 64 GPUs available, you can easily run all your Track A/B experiments simultaneously."*

Concrete patterns:

- One sbatch per (model, layer-band) for activation extraction — e.g. Qwen 3 4B layers 0–9, layers 10–19, layers 20–28 as 3 parallel jobs rather than one job iterating layers.
- One sbatch per (method, cleaning, pool) for any backfill MC — write a generic `run_mc_probes_param.sbatch` that takes `--method --cleaning --pooling` and submit N copies, not a single job that loops.
- For Qwen-Scope SAE encoding (Phase E1): one sbatch per (Qwen3 size, SAE layer). Qwen-Scope ships 14 SAEs across 7 model variants; per the parallelism policy, that's up to 14 simultaneous CPU jobs, not one queue.
- Standard sbatch boilerplate from `v_1/src/cluster/README.md`: partition `voltagepark`, `--gres=gpu:1` (untyped — `gpu:H100:1` is rejected), conda env `thesis`, repo at `~/projects/lititure-review`, 24h walltime by default (Round 2 confirmed no queue penalty).

The cluster-job-runner subagent (`.claude/agents/cluster-job-runner.md`) is the right delegate for writing these scripts. Default behaviour: ask Yarin which parallelism granularity he wants before drafting.

---

## Dependency graph and parallelism plan

This section is the authoritative parallelism contract. Every phase below tells you (i) what it depends on, (ii) what runs in parallel inside it, and (iii) what can start *speculatively* before its prerequisites finish. The policy is **maximum sbatch fan-out**: anything that can be a separate job is a separate job, and we only block when the next step truly cannot start without the previous step's output.

### Dependency DAG

```
Round 2 close (job 8477 aggregated)
        │
        ▼
   Phase 0  (cluster ls, ~10 min)
        │
        ├──► Phase A  (1 hr local; needs Thalesian cuneiBase tier0/mean only)
        │       │
        │       └─[gate]─► Phase B  (parallel sbatch CPU fan-out)
        │                       │
        │                       ├──► Phase C  (parallel sbatch per ruler / per archive)
        │                       │       │
        │                       │       └─[gate]─► Phase C masked extraction (parallel GPU sbatch per method)
        │                       │
        │                       └──► Phase D  (single local job; uses B's best (method, layer))
        │                                │
        │                                └──► Phase E1 attribution analysis (parallel CPU sbatch per layer)
        │
        ├──► (speculative) Random-Qwen re-extraction  (1 GPU sbatch, ~30 min) — runs while Phase A/B unfolds
        │
        └──► (speculative) Phase E1 Qwen3 activation extraction (parallel GPU sbatch per (size, layer-band))
                 │
                 └──► Phase E1 Qwen-Scope SAE encoding (parallel CPU sbatch per (size, sae_layer))
                          │
                          └──► join with Phase D output ──► Phase E1 attribution analysis (above)
                                   │
                                   └──► Phase E2 residualization (parallel CPU sbatch per mask config)
```

### Hard wait points (only these block)

| Wait | Reason | Estimated block time |
|---|---|---|
| Round 2 → Phase 0 | Don't touch the codebase while job 8477 is writing | 0 — Round 2 is closed first by precondition |
| Phase 0 → Phase A | A needs a verified path to Thalesian cuneiBase activations | ~10 min |
| Phase A → Phase B | B is expensive; if A fails its gate, no point running B | ~1 hr |
| Phase B → Phase C | C needs `geodesic_best_layers.json` to know which (method, layer) to LORO | ~half day if B runs local; ~30 min if B is fanned out across cluster CPU sbatch |
| Phase B → Phase D | D needs B's best (method, layer, cleaning, pool) to plot the centroid+spline | same as above |
| Phase D → Phase E1 attribution | E1's geodesic-direction decomposition needs Phase D's curve | ~half day |
| Phase E1 → Phase E2 | E2 needs the Φ_time / Φ_ruler taxonomy from E1 | ~3 days if E1 cluster runs are well-parallelized |

### Speculative parallel work (start without waiting)

These start as soon as Phase 0 confirms inventory, even though their downstream consumer hasn't run yet. Worst case: a gate fails and we throw the work away. Best case (and most likely): we save days of wall-clock by overlapping cluster jobs with local analysis.

1. **Random-Qwen re-extraction** (if Phase 0 needs it): one GPU sbatch, ~30 min. Runs while Phase A is happening on Mac. No dependency.
2. **Qwen 3 4B activation extraction** for Phase E1: 3 parallel GPU sbatch (layer bands 0–9, 10–19, 20–end). Starts as soon as Phase 0 finishes. Whether we *use* it depends on Phase A's gate, but extraction is cheap enough (~few hours each on H100) that pre-launching is positive expected value.
3. **Qwen 3 14B activation extraction**: 3–4 parallel GPU sbatch with more memory. Defer until Phase B confirms there's a real geodesic signal — extraction at 14B is ~6× more expensive than 4B and we don't want to throw it away if Phase A fails.
4. **Qwen-Scope SAE encoding** for Qwen 3 4B: starts as soon as 4B activations are written. Up to ~7 parallel CPU sbatch (one per SAE layer per model variant). Pure matmul, no GPU needed.

### Intra-phase parallelism (per phase)

Each phase below has a `### Parallelism` subsection that names the exact sbatch fan-out granularity and the maximum simultaneous job count. The summary:

| Phase | Granularity | Max simultaneous jobs | Job type |
|---|---|---|---|
| 0 | none | 1 | cluster `ls` |
| A | none | 1 | local |
| B | one per (method × cleaning × pool); layers iterated inside each job | 16–20 | cluster CPU (preferred) or local processes |
| C LORO | one per (ruler) | 8–17 (≥10-fragment rulers) | cluster CPU |
| C masked extraction (gated) | one per (method, cleaning) | 8 | cluster GPU |
| D | one per (method to plot) | up to 5 | local |
| E1 extraction | one per (Qwen3 size, layer band) | 3–6 (4B alone) or 6–10 (4B + 14B) | cluster GPU |
| E1 SAE encoding | one per (Qwen3 size, sae layer) | up to 14 (Qwen-Scope's full layer set) | cluster CPU |
| E1 attribution | one per (Qwen3 size, sae layer) | same as above | cluster CPU |
| E2 residualization | one per (mask config: hard/soft × Φ_time-only / inverse-mask / etc) | 4–8 | cluster CPU |

### Default fan-out granularity per phase

For each phase that needs sbatch, the default fan-out is the **finest** granularity in the table above, unless Yarin overrides. Concretely: when the cluster-job-runner subagent is invoked to write Phase B scripts, it should produce 20 individual `.sh` files (one per method × cleaning × pool combo), not a single looping script. Same for Phase C and E1.

---

## Hypothesis Ladder

| Phase | Question | Cost | Output |
|---|---|---|---|
| **0** | Pre-flight: are all required activations on disk? Random-Qwen path verified? | ~10 min cluster `ls` + 0 cluster compute | activation inventory JSON; go/no-go for Phase B |
| **A** | Does the geodesic readout exist and beat linear on the current Thalesian best layer? | ~1 hr local | single-layer proof-of-concept; go/no-go for Phases B–D |
| **B** | Full layer × method × pool × cleaning scoreboard with geodesic metrics | ~half day local | `geodesic_best_layers.json`; linear-vs-geodesic comparison plot |
| **C** | Is the recovered manifold genuinely temporal (not ruler/archive shortcut)? | ~1 day local + optional parallel sbatch for masked extraction | leave-one-X-out drop tables; honesty plots |
| **D** | Goodfire-style centroid+spline visualization + thesis-figure pass | ~half day local | bin-centroid plot in 3D PCA with non-periodic cubic spline; per-method comparison panel |
| **E1** | SAE feature attribution along the geodesic (interpret) — Qwen-Scope on Qwen3 | ~3 days parallel cluster | which SAE features carry temporal vs ruler vs genre signal |
| **E2** | SAE feature ablation along the geodesic (clean) — reconstruct h' from temporal features only, re-fit geodesic | ~2 days local + parallel cluster | causal-style "the temporal direction is isolable and ruler-independent" claim |

Phases 0, A–D are **local laptop work** (Phase 0 is one cluster `ls`). Activations live in NPZ files; PCA, kNN, Isomap, splines are sklearn on at most (1193, 4096)-sized matrices. No GPU, no Slurm, no cluster races for the core analysis. Phases E1/E2 are the cluster-heavy half.

---

## Phase 0 — Pre-flight inventory check

**Hypothesis 0:** All activations needed for Phases A–D exist on disk somewhere on the cluster, regardless of whether they're at the path `run_mc_probes.py` originally expected.

### Procedure

1. Yarin runs the three-path `ls` from the Data Inventory section. Result goes into `v_1/src/geodesic/results/phase_0_inventory.json` with one entry per (method, cleaning, pool, path-found).
2. For any (method, cleaning, pool) missing from all three paths: decide between symlink fix, re-extract, or drop-from-scope (Random-Qwen is the canonical case).
3. Confirm `orcc_corpus.parquet` is the version with 1,193 year-labeled fragments (`canonical_sizes` memory check).
4. Spot-check one NPZ per method to confirm shape `(n_fragments, d_model)` and dtype `float16` or `float32`.

### Gate

- **Proceed to Phase A if:** Thalesian cuneiBase-400m tier0/mean activations are confirmed on disk (the minimum viable input for Phase A).
- **Block Phase B until:** all 4 non-Random methods have all (cleaning, pool) combos confirmed, OR a decision is recorded in `phase_0_inventory.json` to skip a missing combo.
- **Random-Qwen specifically:** either symlink-fix today, schedule re-extraction as one cluster job, or write a one-line note in the Round 3 report that "Random-Qwen geodesic results omitted pending activation re-extraction." All three are acceptable.

### Deliverable

`v_1/src/geodesic/phase_0/inventory.py` (or just a notebook) + `phase_0_inventory.json` listing what's on disk and where. ~10 min of Yarin's cluster-terminal time.

### Parallelism

Single command; nothing to fan out. **As soon as Phase 0 finishes, launch the speculative parallel work** listed in the Dependency graph section: Random-Qwen re-extraction (if needed) + Qwen 3 4B activation extraction (3 sbatch). Yarin can paste both blocks back-to-back; they run on different nodes.

### Depends on

Round 2 close (job 8477 aggregated → report finalized → committed). Nothing else.

---

## Phase A — Single-layer proof of concept

**Hypothesis A:** On Thalesian cuneiBase-400m, layer 12, tier0 cleaning, mean pooling — the configuration that won Round 2's year regression — the geodesic 1D coordinate beats PLS in pairwise-order accuracy and Spearman.

### Procedure

1. Load `activations/thalesian_cunei400m_tier0_mean/layer_12.npz` and ORCC parquet `year` column. Drop rows with missing year. Final `X ∈ R^{n×d}`, `y ∈ R^n`.
2. Apply the advisor's pipeline:
   - Center: `X_c = X − X.mean(axis=0)`
   - PCA to `min(64, n-1)` components
   - L2-normalize each row
   - Build smallest-connected kNN cosine graph (`k` sweep 3..50, accept first connected)
   - Compute two 1D coordinates:
     - **A1 (Isomap):** `Isomap(n_neighbors=k, n_components=1, metric="cosine")`
     - **A2 (earliest-bin geodesic):** average shortest-path distance from all texts whose lower-bound year falls in the earliest 100-yr bin
   - Sign-flip if Spearman with year is negative
3. Compute three evaluation metrics:
   - Spearman(coord, year)
   - Pairwise-order accuracy with 100-yr margin (advisor's main metric)
   - Temporal neighbor purity at k=10 with ±100-yr window, plus 500-permutation null
4. Compare to the existing PLS L12 numbers from `pls_results_thalesian_cunei400m.json` (Spearman 0.467, R² +0.105, MAE 75 yr). Pairwise-order acc must be recomputed for PLS on the same labels for apples-to-apples.

### Gate (locked here, not after seeing numbers)

- **Proceed to Phase B if:** either Isomap or earliest-bin geodesic improves Spearman by ≥0.05 over PLS, OR pairwise-order accuracy ≥0.70.
- **Stop and report null if:** both geodesic coordinates underperform PLS by ≥0.05 Spearman AND pairwise-order accuracy <0.60. In that case, Round 3 becomes a one-page negative result and we proceed to Phase 2 (scale) from the Round 2 plan instead.

### Deliverable

`v_1/src/geodesic/phase_a/poc_thalesian_L12.ipynb` (or `.py`) + a one-table markdown summary appended to `orcc_round2_REPORT.md` under a new "Round 3 — Geodesic Pilot" section.

### Parallelism

Single layer, single notebook — no fan-out. **While Phase A runs locally on Mac, the speculative Qwen 3 4B extraction sbatch jobs continue in the background.** When Phase A gate clears, the Qwen 3 activations are likely already partially extracted, shaving hours off Phase E1's wall-clock.

### Depends on

Phase 0 (path confirmation for Thalesian cuneiBase tier0/mean activations only).

---

## Phase B — Layer × Method scoreboard

**Hypothesis B:** Across all five methods, the *best geodesic layer* differs from the *best linear PLS layer* for at least two methods, and Thalesian's lead over Qwen/MLM grows when measured geodesically.

### Procedure

1. For every (method ∈ {qwen, random_qwen, mlm_aeneas, thalesian_akk300m, thalesian_cunei400m}, cleaning ∈ {tier0, maximal}, pool ∈ {mean, last}, layer ∈ all_available):
   - Activation loader uses the same dual-path fallback added during Round 2 Phase 3 (`orcc_round1/activations/` → `orcc__embed/activations/`), with an extra check for `activations/qwen2.5-7b-instruct-random/tier0/`. If the loader returns `None`, log as `skipped: <reason>` rather than crash. Random-Qwen tier0/mean is the canonical fail-soft case.
   - Run the Phase A pipeline (PCA64 → L2 → kNN → Isomap 1D + earliest-bin geodesic).
   - Compute Spearman, pairwise-order acc (100-yr margin), temporal neighbor purity, neighbor-purity permutation lift.
2. Aggregate into a single parquet `geodesic_layer_scoreboard.parquet` (~1000 rows; method × cleaning × pool × layer × metric).
3. For each method, pick `geodesic_best_layer` by pairwise-order accuracy (advisor's primary metric). Save as `geodesic_best_layers.json` — same schema as `pls_best_layers.json` but indexed by the geodesic metric.
4. Produce two plots:
   - **B1 — Layer scan per method:** x=layer, y=pairwise-order acc, one line per method (best cleaning/pool only). Two-panel: linear PLS pairwise acc vs geodesic pairwise acc.
   - **B2 — Method comparison at best layer:** bar chart of geodesic pairwise acc per method, with linear PLS pairwise acc overlaid as ghost bars.
5. Permutation baseline: shuffle `year` 500×, recompute pairwise-order acc on shuffled labels per (method, best-layer). Report observed vs null mean ± std. Anything within 2σ of shuffled null is reported as non-significant regardless of absolute value.

### Gate

- **Proceed to Phase C if:** at least one method has geodesic pairwise acc ≥0.65 AND >3σ above its shuffled null.
- **Adjust scope if:** only Thalesian methods clear the bar — Phase C runs on Thalesian only; Qwen/MLM/Random reported as "no geodesic structure detected" controls.

### Deliverables

- `v_1/src/geodesic/phase_b/scan.py` (CLI: `--methods qwen,thalesian_cunei400m,... --cleanings tier0 --pools mean`)
- `v_1/src/geodesic/results/geodesic_layer_scoreboard.parquet`
- `v_1/src/geodesic/results/geodesic_best_layers.json`
- Two PNG plots in same results dir
- Markdown table appended to Round 2 report

### Risks

- **PCA dimensionality:** advisor suggests PCA64 throughout. For models with `d_model=896` (Thalesian UMT5) this is aggressive; for Qwen `d_model=3584` very aggressive. Sweep `n_pca ∈ {32, 64, 128, 256}` on the Phase A winner to confirm 64 is reasonable before locking. Document choice.
- **kNN k selection:** advisor's "smallest connected" rule can be brittle when there's a tight cluster + outlier. If `k_max=50` is hit without connectivity, fall back to symmetric kNN with mutual=False (already done in advisor's code), then log the disconnect.
- **L0 degeneracies:** Round 2 hit rank-deficient SVD crashes on L0 PLS. PCA → Isomap is more stable but still wrap in try/except; report NaN rather than skip silently.

### Parallelism

**Default fan-out: 20 parallel cluster CPU sbatch jobs**, one per (method × cleaning × pool). Each job iterates layers internally (typically ~29 layers for Qwen-class, ~13 for Thalesian UMT5). Job script template:

```
sbatch v_1/src/geodesic/phase_b/sbatch/scan__<method>__<cleaning>__<pool>.sh
```

Total: 5 methods × 2 cleanings × 2 pools = 20 scripts. Submit all 20 with one `for` loop; cluster runs them on 20 different CPU nodes (no GPU needed — Phase B is sklearn). If queue depth is a constraint, group as 5 jobs by method (each iterating cleaning × pool × layer internally) — but the 20-way fan-out is default.

**Aggregation step** (`aggregate.py`) waits for all 20 to finish. Use `--dependency=afterok:$JOB1:$JOB2:...:$JOB20` on the aggregator's sbatch, or just have Yarin paste the aggregation command after `squeue -u $USER` shows empty.

**Why not local on Mac:** the original plan said "half day local." That's still possible if you want zero cluster involvement. But fanning out to 20 cluster CPU sbatch finishes in ~10–20 min wall-clock instead, which is the better default given user preference for parallelism.

### Depends on

Phase 0 (inventory complete) + Phase A (gate passed). Speculative Qwen 3 extraction can keep running in parallel.

---

## Phase C — Honesty pass

**Hypothesis C:** The recovered temporal manifold survives leave-one-ruler-out evaluation with a pairwise-order accuracy drop of less than 0.10 absolute. If the drop exceeds 0.20, the manifold is partially a ruler-cluster trajectory and should be reported as such.

### Procedure

1. **Leave-one-ruler-out (LORO) on geodesic coordinate.** For each ruler with ≥10 fragments: hold out all fragments of that ruler, refit PCA + kNN + geodesic on the remaining data, project held-out fragments through the fitted PCA + Isomap, score pairwise-order accuracy of held-out fragments against held-in fragments (only cross-ruler pairs, with 100-yr margin).
2. **Leave-one-archive-out / leave-one-genre-out** if archive/genre metadata is available in ORCC parquet — check.
3. **Honesty plot:** the Phase D centroid plot but colored by ruler instead of year. If ruler clusters dominate the 3D structure, mark the figure as "shortcut warning."
4. **Optional — masked text robustness:** if Phase B shows strong Thalesian signal, request a cluster job to re-extract activations with `[RULER]` and `[YEAR]` tokens substituted in the input text. This is the only step in Round 3 that may need GPU. Defer decision until after Phase B.

### Gate

- **Strong claim allowed if:** LORO drop <0.10 AND ruler-colored centroid plot does not form ruler-islands.
- **Hedged claim if:** LORO drop 0.10–0.20. Report as "temporal signal partially confounded with ruler identity; manifold remains informative beyond ruler labels."
- **Null claim if:** LORO drop >0.20. Report as "what looked like a temporal manifold is largely ruler-cluster geometry."

### Deliverables

- `v_1/src/geodesic/phase_c/loro.py`
- `v_1/src/geodesic/results/loro_robustness.json`
- Honesty plot PNG + markdown narrative for Round 3 report

### Parallelism

**LORO: one cluster CPU sbatch per ruler.** ORCC has 8–17 rulers with ≥10 fragments (subset depends on cutoff). Each LORO job refits PCA + kNN + Isomap on the remaining ~1100 fragments — sklearn CPU-only, ~5 min per ruler. Fan out: 8–17 simultaneous sbatch jobs finish in ~10 min instead of 1–2 hours sequential.

Job template:
```
sbatch v_1/src/geodesic/phase_c/sbatch/loro__<ruler_id>.sh
```

For leave-one-archive-out and leave-one-genre-out (if metadata available), same fan-out pattern. Total: up to ~30–40 simultaneous LORO sbatch jobs across ruler + archive + genre.

**Masked-text extraction (gated, optional): one GPU sbatch per (method, cleaning) on the Phase B winner method only.** Typically 1 method × 2 cleanings = 2 parallel GPU sbatch. Defer dispatch until Phase B verdict is in.

### Depends on

Phase B (`geodesic_best_layers.json`). LORO needs B's chosen (method, layer, cleaning, pool) to know which configuration to refit. Masked-text extraction is gated on B's signal strength.

---

## Phase D — Goodfire-style centroid + spline visualization

**Hypothesis D:** The 100-year bin centroids of the best-performing (method, layer, cleaning, pool) trace a smooth path in 3D PCA, with no centroid more than two bin-widths away from its chronological neighbor.

### Procedure

1. Use Phase B's `geodesic_best_layer` for the chosen method.
2. Bin fragments into 100-yr lower-bound bins. Require ≥5 fragments per bin; drop sparse bins (advisor's threshold).
3. Per bin, compute centroid in PCA64 space; L2-normalize centroid.
4. Fit non-periodic UnivariateSpline (cubic) per PCA dimension, weighted by `√bin_count`, smoothing `s = len(t) * 0.001` as starting point. Sample 300 points along year axis.
5. Project both individual texts and centroids to 3D PCA for plotting. Project the sampled spline curve through the same PCA3 transform.
6. Produce four versions of the plot (advisor's "version I would trust most"):
   - D1: unmasked text, colored by year (the headline visual)
   - D2: same plot colored by ruler
   - D3: same plot colored by archive (if metadata available)
   - D4: if Phase C masked extraction ran, masked-text version colored by year
7. Compute arc-length along the fitted spline; report Spearman(arc_length, bin_center_year). This is the "is the curve really 1D and ordered" check.

### Deliverables

- `v_1/src/geodesic/phase_d/centroid_spline.py`
- 4× PNG (D1–D4) at publication resolution
- Arc-length-vs-time Spearman in the Round 3 report table

### Risks

- **Sparse bins:** ORCC year distribution is skewed toward Neo-Assyrian. Some early bins may have <5 fragments and get dropped. Plot will visibly start at ~9th century BCE. Document this honestly.
- **Spline overfitting:** if `s=0` (interpolation), curve wiggles between sparse early bins. Sweep `s ∈ {0, 0.001n, 0.01n, 0.1n}` and pick by eye + arc-length-vs-time Spearman.

### Parallelism

**Up to 5 parallel local processes**, one per method (qwen / random_qwen / mlm_aeneas / thalesian_akk300m / thalesian_cunei400m), if you want the centroid plot for all methods rather than just the Phase B winner. Default: only plot the Phase B winner method (1 process). The four sub-plots D1–D4 within a single method are sequential (they share the same fitted PCA + spline).

### Depends on

Phase B's `geodesic_best_layers.json`. D2 (ruler-colored) and D3 (archive-colored) also need ORCC parquet metadata which is already on disk.

---

## Phase E — SAE attribution AND signal cleaning via Qwen-Scope (replaces Round 2 Phase 2)

This phase absorbs three previously-separate threads into one coherent post-Phase-D module:
- The Round 2 Phase 2 (scale) sweep on Qwen 3 dense {4B, 14B, optional 32B}.
- The Track C SAE interpretation plan (`v_1/src/sae/plan/PLAN.md`), originally written for Arditi's Qwen 2.5-7B 131k SAE but now retargeted to Qwen-Scope.
- A new SAE-as-cleaner / causal-mediator analysis that was explicitly excluded from the original Track C plan (line 527: "Do NOT implement Feldman's residualization / CATE — doesn't apply to our classification task"). That exclusion was correct for the letters period-classification task; it is *no longer* correct for Round 3's year regression + manifold story. Phase E2 reintroduces residualization with that justification recorded.

### Why Qwen-Scope, not Arditi

The Qwen-Scope paper (`papers/txt/Geometric Representation papers/Qween-Scope.txt`, dated 2026-04-30) released 14 SAEs across 7 model variants from Qwen3 and Qwen3.5, both dense and MoE. This is the "official SAE" the Round 2 Phase 2 plan asked us to "verify availability" of — it now exists. Switching from Arditi 131k on Qwen 2.5-7B to Qwen-Scope on Qwen3 gives us (a) layer-wise coverage across the whole stack, not just L7/15/23; (b) multiple model sizes, so the scale sweep and SAE attribution become the same experiment; (c) the paper's own steering / classification toolkit as a starting point for E2.

The Track C `PLAN.md` should be updated to: (i) flip the `do NOT implement Feldman residualization` line to a conditional `OK for year-regression manifold task`, (ii) add a Qwen-Scope variant of `01_extract_sae_features.py` that loads Qwen-Scope SAEs instead of Arditi.

### CRITICAL — Pooling constraint: SAEs require last-token, not mean

**Qwen-Scope SAEs were trained on per-token residual stream activations**, identical to the Arditi pooling constraint already noted in `v_1/src/sae/plan/PLAN.md` lines 64–75. Mean-pooled vectors are synthetic combinations the SAE never saw in training; encoding them produces meaningless feature activations. **All Phase E activations are last-token pooled.** This is non-negotiable and breaks symmetry with Phases A–D, which used mean-pooling on the Thalesian winner.

This pooling change has three downstream consequences that Phase E1 must handle explicitly:

1. **The "Phase B geodesic" used for attribution in step 4 below is NOT the same geodesic as Phases B/C/D.** Phases B/C/D's headline geodesic is on Thalesian cuneiBase-400m tier0/mean. Phase E1's attribution geodesic is on Qwen 3 (size X) tier0/last. These are different representations of different models; the geodesics are not directly comparable.

2. **Phase E1 must compute its own last-token geodesic on Qwen 3 first** before attributing it to SAE features. This adds a sub-step (E1.0 below) that wasn't in the original phase description.

3. **The cross-method comparison stays valid at the geodesic-existence level** — "does Qwen 3 last-token also produce a temporal geodesic, and does scale make it stronger?" — but you cannot claim "Qwen-Scope features explain Thalesian's manifold." The Qwen-Scope features explain Qwen 3's manifold. Cross-model SAE alignment is a separate (harder) research question, not in scope for Round 3.

### Phase E1 — Interpret (the original Track C goals on the new substrate)

For each (Qwen3 size ∈ {4B, 14B, optional 32B}, SAE layer per Qwen-Scope's layer set):

1. **E1.0 — Extract Qwen 3 last-token activations** on ORCC at the SAE's hook layer. **Pooling = last_token** (NOT mean). Parallel sbatch fan-out per (model_size, layer-band) — one sbatch script per band, ~6 simultaneous jobs for the 4B + 14B combo. Cleaning = tier0 (matches everything else in Round 3).
2. **E1.1 — Fit Qwen 3 last-token geodesic.** Run the Phase A/B pipeline (PCA64 → L2 → kNN → Isomap 1D + earliest-bin geodesic) on Qwen 3's last-token activations at each Qwen-Scope SAE layer. Compute pairwise-order accuracy + Spearman vs year. This gives Qwen 3's own geodesic coordinate — distinct from Thalesian's. **Why this step exists:** SAE attribution in step 4 needs a geodesic coordinate that lives in the same activation space as the SAE features. The Thalesian geodesic from Phase B lives in Thalesian's activation space and cannot be attributed to Qwen-Scope features.
3. **E1.2 — Run Qwen-Scope SAE encoder** on those last-token activations (CPU-bound matmul; one sbatch per SAE layer). Output: sparse feature matrix `z ∈ R^{n × n_features}`.
4. **E1.3 — Sparse probe accuracy curve:** for each layer, sparse-probe pairwise-order accuracy on year using top-k SAE features (k ∈ {16, 64, 256, 1024}). Compare to (a) Qwen 3's dense linear probe at that layer, (b) Qwen 3's last-token geodesic from E1.1.
5. **E1.4 — Geodesic-direction decomposition:** E1.1 produced a 1D Qwen 3 geodesic coordinate per fragment. Train a quick logistic / ridge model from SAE features → that coordinate. Top-aligned features = "what Qwen 3's timeline is made of."
6. **E1.5 — Per-bin feature profiles:** for each 100-yr bin (definition from Phase D, but applied to Qwen 3's geodesic now), which SAE features have highest mean activation? Compare adjacent bins to find features that flip on/off across temporal boundaries.
7. **E1.6 — Cross-layer / cross-size comparison:** do the same top features appear at multiple layers and sizes? Scale-invariant features are the strongest interpretation candidates.
8. **E1.7 — Automated feature interpretation:** differential bigram analysis on high vs low-activation fragments per top feature (reuses Track C Analysis E pipeline).
9. **E1.8 — Cross-model bridge (optional, soft claim only):** for each top temporally-aligned Qwen-Scope feature, mean-activate it across all fragments and check whether that mean activation correlates with the Thalesian geodesic coordinate from Phase B. This is a weak cross-model alignment test; report Spearman with appropriate hedging.

### Phase E2 — Clean (SAE residualization on the manifold)

Hypothesis: if we reconstruct h' from only the SAE features identified in E1 as temporally aligned, and zero out features identified as ruler / archive / genre aligned, then:
- The **Qwen 3 last-token geodesic** (the E1.1 geodesic) pairwise-order accuracy on h' is ≥ accuracy on h (signal preserved or improved).
- A leave-one-ruler-out pass on h' shows a smaller drop than on h (confound reduced).

**All E2 quantities are last-token on Qwen 3**, same pooling regime as E1 — never mix in Thalesian or mean-pooled activations here. The "h vs h'" comparison is Qwen 3 dense vs Qwen 3 SAE-reconstructed, both last-token.

Procedure:

1. **Feature taxonomy.** From E1.4/E1.5, partition SAE features into Φ_time (top-aligned with E1.1's Qwen 3 last-token geodesic coordinate), Φ_ruler (top-aligned with ruler identity via separate classifier trained on the same SAE features), Φ_archive (likewise for archive), Φ_other.
2. **Reconstruct h'** = SAE_decode(z ⊙ mask), where `mask` zeros out Φ_ruler ∪ Φ_archive and keeps Φ_time ∪ Φ_other. Sweep several mask strengths (hard zero, soft attenuation 0.5×, 0.1×) to avoid all-or-nothing.
3. **Re-fit the Qwen 3 last-token geodesic on h'** (same PCA → kNN → Isomap pipeline as E1.1), compute pairwise-order acc + Spearman + neighbor purity.
4. **Re-fit a Qwen-3-flavored LORO on h'** (analogous to Phase C but on Qwen 3's last-token activations), compute drop.
5. **Compare h vs h'** at the centroid level (re-bin Qwen 3's geodesic into 100-yr bins, fit a Qwen-3-flavored spline): do bins move closer along the spline? Does arc-length-vs-time Spearman improve?
6. **Sanity reverse:** also run the inverse mask (keep ONLY Φ_ruler ∪ Φ_archive, zero Φ_time). Geodesic readout on this reconstruction should DROP, confirming that we actually separated the features cleanly. If both reconstructions preserve the geodesic, the decomposition didn't work and we report that as a negative result.

**Note on apples-to-apples claims:** the strongest claim E2 can support is *"on Qwen 3 last-token activations, residualization with Φ_time-only SAE features preserves the temporal manifold while reducing ruler-confound."* A claim like *"residualization improves Thalesian's manifold"* is NOT supported because Thalesian doesn't have a Qwen-Scope SAE. Report E2 results as a Qwen-3-internal causal-mediator finding.

### Gates

- **E1 proceeds if:** Phase D succeeded for the headline (Thalesian mean-pool) story AND Qwen-Scope availability on HuggingFace verified for the chosen Qwen3 size AND Qwen 3 last-token activations confirmed extractable (E1.0 sanity).
- **E1.1 hard prerequisite:** the Qwen 3 last-token geodesic must show *some* temporal signal (pairwise-order acc ≥0.55 at the best Qwen-Scope SAE layer) before E1.2–E1.7 are worth running. If Qwen 3 has no last-token temporal structure, SAE attribution has nothing to attribute — report E1 as "no Qwen 3 last-token timeline found, SAE attribution skipped" and proceed to E2 with the Φ_time set empty (E2 then becomes a vacuous null).
- **E1 strong claim:** top-k SAE sparse probe with k ≤ 64 matches the Qwen 3 dense linear probe (last-token, same layer) within 0.05 pairwise-order acc.
- **E2 proceeds if:** E1 produces a Φ_time set of ≥10 features with clear differential bigram interpretations.
- **E2 strong claim:** h' Qwen-3 last-token geodesic accuracy ≥ h accuracy AND h' LORO drop < h LORO drop by ≥0.05 AND inverse-mask reconstruction shows accuracy collapse.

### Cost

- Phase E1: 1 day for Qwen 3 4B activation extraction (parallel sbatch across layer bands) + 1 day for Qwen-Scope SAE encoding (parallel sbatch per SAE layer) + 1 day for analysis. ~3 days total wall-clock with good parallelism.
- Phase E2: ~1 day (no new extraction — all matrix ops on E1 outputs).
- 14B sweep: add another 2 days; 32B is optional and only if 4B vs 14B shows a positive scaling trend.

### Parallelism

Phase E is the cluster-heaviest part of Round 3 and the place where aggressive sbatch fan-out saves the most wall-clock. Breakdown by sub-step:

**E1 extraction — Qwen 3 activations (GPU):**
- One sbatch per (Qwen3_size, layer_band).
- 4B minimum: bands {0–9, 10–19, 20–end} = 3 parallel GPU sbatch.
- 4B + 14B: 6 simultaneous GPU sbatch.
- 4B + 14B + 32B: 9 simultaneous GPU sbatch (32B needs 2 GPUs per job → uses 18 GPUs total; still well within the 64-GPU budget).
- Can start **speculatively** right after Phase 0, in parallel with Phase A/B/C/D local work.

**E1 SAE encoding — Qwen-Scope (CPU):**
- One sbatch per (Qwen3_size, sae_layer). Pure matmul, no GPU.
- Qwen-Scope ships up to 14 SAEs across the suite; for any one Qwen3 size, expect ~3–5 SAE layers covered (verify on HF).
- 4B with 5 SAE layers = 5 parallel CPU sbatch. 4B + 14B = 10 parallel CPU sbatch. With 832 CPUs available, queue depth is not a constraint.
- Depends on E1 extraction for that (size, layer); use `--dependency=afterok:$EXTRACT_JOB` so each SAE encode job auto-launches when its activations are written.

**E1 attribution analysis (CPU):**
- One sbatch per (Qwen3_size, sae_layer) for sparse probing + geodesic decomposition + bin profiles.
- Same fan-out as SAE encoding; same `--dependency=afterok` pattern.
- Depends on Phase D's curve (centroid coordinates) — block until D finishes, then submit.

**E2 residualization (CPU):**
- One sbatch per (mask_config). Configs: hard-zero Φ_ruler+Φ_archive, soft 0.5×, soft 0.1×, inverse-mask sanity reverse.
- 4–8 parallel CPU sbatch. All matrix ops on already-encoded SAE features — no new extraction needed.
- Depends on E1 attribution to know the Φ_time / Φ_ruler / Φ_archive partition.

**Dependency chain for the E1+E2 cluster pipeline:**
```
Phase 0 done
   │
   ├──► [GPU sbatch × 3]  extract Qwen3-4B layer-bands 0-9, 10-19, 20-end
   │       │
   │       └──► [CPU sbatch × ~5]  Qwen-Scope SAE encode (afterok dependency)
   │                │
   │                └──► wait for Phase D ──► [CPU sbatch × ~5]  E1 attribution
   │                                                  │
   │                                                  └──► [CPU sbatch × 4–8]  E2 residualization
   │
   ├──► [GPU sbatch × 3]  extract Qwen3-14B (same pattern, optional, if 4B shows signal)
   │
   └──► [GPU sbatch × 3]  extract Qwen3-32B (same pattern, optional, if 14B shows positive scaling)
```

Submit the 4B extraction chain immediately after Phase 0. Submit the 14B and 32B chains after Phase B's geodesic verdict (no point extracting 14B if 4B shows nothing).

### Deliverables

- `v_1/src/sae/qwen_scope/01_extract_qwen3_activations.py` (parametrized by size + layer band)
- `v_1/src/sae/qwen_scope/02_encode_sae.py` (one sbatch per SAE layer)
- `v_1/src/sae/qwen_scope/03_attribute.py` (E1: sparse probing + geodesic decomposition + bin profiles)
- `v_1/src/sae/qwen_scope/04_residualize.py` (E2: mask, reconstruct, re-fit geodesic + LORO)
- Updated `v_1/src/sae/plan/PLAN.md` reflecting Qwen-Scope substrate and the E2 unblock
- `v_1/src/geodesic/results/sae_attribution_E1.json`, `sae_residualization_E2.json`
- Plots: top-feature differential bigram charts; h vs h' geodesic comparison; LORO drop comparison; scaling plot across Qwen3 sizes

---

## Where every new file lives

| Purpose | Path |
|---|---|
| Phase 0 inventory | `v_1/src/geodesic/phase_0/inventory.py` + `results/phase_0_inventory.json` |
| Phase A POC | `v_1/src/geodesic/phase_a/` |
| Phase B layer scan driver | `v_1/src/geodesic/phase_b/scan.py` |
| Phase B aggregator | `v_1/src/geodesic/phase_b/aggregate.py` |
| Phase C LORO | `v_1/src/geodesic/phase_c/loro.py` |
| Phase D centroid plot | `v_1/src/geodesic/phase_d/centroid_spline.py` |
| Phase E1 Qwen-Scope extract + encode + attribute | `v_1/src/sae/qwen_scope/01_extract_qwen3_activations.py`, `02_encode_sae.py`, `03_attribute.py` + sbatch siblings |
| Phase E2 residualize | `v_1/src/sae/qwen_scope/04_residualize.py` |
| Shared utilities (PCA, kNN, metrics) | `v_1/src/geodesic/utils.py` |
| Results root | `v_1/src/geodesic/results/` |
| Round 3 report | `v_1/src/geodesic/results/orcc_round3_REPORT.md` (mirrors round2_REPORT.md style) |
| Updated Track C plan | `v_1/src/sae/plan/PLAN.md` (Qwen-Scope substrate; E2 unblock note) |

The `v_1/src/geodesic/` and `v_1/src/sae/qwen_scope/` directories do not yet exist; Phase 0 and Phase E1 create them respectively.

---

## Sequencing relative to Round 2

1. **Round 2 must close first.** Job 8477 lands → `aggregate_p0.py` rerun → Round 2 report finalized → commit + push. Do not start Round 3 before that.
2. **Phase 0 (cluster `ls`) starts immediately after**, paid for in 10 minutes of Yarin's terminal time.
3. **Phase A** starts the same day Phase 0 closes (1-hr local task).
4. The original Round 2 Phase 2 (scale on Qwen 3) is **absorbed into Phase E1** of this plan, not run separately. It is informed by Phase B's geodesic results: if Thalesian's geodesic lead over Qwen pretrained is dramatically larger than the linear lead, the case for scaling Qwen weakens (architecture/training-data choice dominates capacity) — we'd do 4B only as a confirmatory control rather than the full 4B/14B/32B sweep. If they match, scale Qwen 3 fully per Phase E1 as planned.
5. SAE work sequences **after** Phase D, not in parallel. Phase D's curve is what Phase E1 interprets.

---

## What "done" looks like for Round 3

A single `orcc_round3_REPORT.md` with:
1. The motivation: linear probing can systematically miss curved temporal structure.
2. Phase 0: data inventory table; Random-Qwen disposition recorded.
3. Phase A pilot: yes/no on the geodesic improvement at the Round 2 best layer.
4. Phase B: full layer × method scoreboard; linear-vs-geodesic comparison; layer-selection-bias quantified.
5. Phase C: LORO drop table; honesty plot; verdict on ruler-confound.
6. Phase D: the four centroid+spline plots; arc-length-vs-time Spearman.
7. Phase E1 (if reached): Qwen-Scope sparse probe accuracy curve; top temporally-aligned features per layer + size; differential bigram interpretations; scaling plot across Qwen3 sizes.
8. Phase E2 (if reached): h vs h' geodesic comparison; LORO drop on h'; inverse-mask sanity check.
9. One-paragraph recommendation to advisor on the new shape of the thesis chapter.

---

## Cluster constraints

Phases 0, A–D run on Mac (Phase 0 is one cluster `ls`). Phases E1–E2 and the optional Phase C masked extraction need cluster. When cluster jobs are needed:

- Follow the Cluster Parallelism Policy section above: fan out over many small sbatch jobs.
- Partition: `voltagepark`, `--gres=gpu:1` untyped, conda env `thesis`, repo at `~/projects/lititure-review`, 24h default walltime.
- Yarin runs sbatch; the cluster-job-runner subagent (`.claude/agents/cluster-job-runner.md`) drafts the scripts and the copy-paste command block.
- Auto-push pattern in sbatch tails from Round 2 carries over: `git add … && git commit -m "…" || true && git push origin main || echo "WARNING"`.
- Random-Qwen activation re-extraction (if needed per Phase 0 disposition) reuses `01b_extract_random_baseline.py`.
- Phase C masked-text extraction reuses `round2_phase3/extract_enc_activations.py` as the template.

---

## Final pre-commitments

- **No layer is selected on test data.** Phase B selects on the full corpus only because there is no held-out test split in Round 1/2 — the 100-yr-margin pairwise accuracy doubles as a coarse generalization metric. If Round 3 results justify a Round 4, that round adds a proper grouped train/dev/test split.
- **PLS stays as the linear baseline.** Round 3 does not retract Round 2's PLS numbers; it adds a parallel column. The R1 imbalanced table (38/38 cells) is the ground-truth linear surface; the 12/44 balanced-MC cells stay as-is unless Phase B's winner specifically needs MC backfilling.
- **No spline cheating.** The spline is fitted on centroid year + activation centroid only; year labels are not used to choose PCA dimensions or kNN k.
- **Random-Qwen is not blocking.** If Phase 0 cannot recover Random-Qwen activations within ~30 min cluster time, Round 3 proceeds without it and the gap is footnoted.
- **Honest null is acceptable.** If Phase A fails its gate, Round 3 becomes a one-page negative-result note and we go back to the Round 2 Phase 2 (scale) plan unchanged. The plan must survive this outcome without rewriting Phases B–E retroactively.
- **SAE-as-cleaner unblock is recorded explicitly.** The Track C PLAN.md "Do NOT implement Feldman's residualization" rule applied to period classification on the letters corpus. It does NOT apply to year-regression manifold analysis on ORCC. Phase E2 is allowed; the original `do NOT` line is amended, not deleted, with the year-regression carve-out noted.
