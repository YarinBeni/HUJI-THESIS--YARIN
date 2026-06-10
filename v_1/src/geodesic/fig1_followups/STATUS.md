# Fig-1 follow-ups — STATUS (single source of truth)

One place to track everything from the 03.06.26 advisor review onward. Update
this file as work lands. Detail/findings live in the per-script READMEs and in
memory `project_fig1_followups.md`; this is the checklist + pointers.

_Last updated: 2026-06-10_

---

## The 5 agreed next-steps (from the meeting)

- [x] **Task 3 — error overlap / per-fragment analysis** (see `error_overlap/`). Done across regimes (below).
- [x] **Task 4 — PLS components tradeoff** (see `pls_ksweep/`). Done; selection-bias finding.
- [ ] **Task 1 — closed-model baseline** (GPT-5 / Claude / Gemini via OpenRouter). NOT started.
- [ ] **Task 2 — gpt-oss-120B** (open-weights, keeps activations). RUNNING — jobs 9555 (extract) → 9556 (probe), as step 0 of Task 5.
- [ ] **Task 5 — fine-tune** Qwen / gpt-oss on Akkadian NTP. RUNNING — pre-EDA done (no vocab expansion: `v_1/src/finetune/eda/results/TOKENIZER_EDA.md`), plan `v_1/src/finetune/PLAN_finetune_ntp.md`. Phase-1 jobs submitted 11.06: 9554 (data) → 9557_[0-3] (1.7B pilot, cuts 0/9/19/25) → 9558 (probe+scoreboard). **Gate on 9558's scoreboard before 8B/120b.** Run log: `v_1/src/finetune/README.md`.
- [ ] **Present figs 0/2/3/4/5** to advisors (only Fig 1 shown). Handoff: `yarin/HANDOFF_fig_presentation.md`.

## Task 4 — PLS k-sweep  (`pls_ksweep/`)
- [x] Balanced k-sweep 1..64 (k=128 above the ~126 balanced ceiling). Temporal signal ~2–3 dim.
- [x] Selection-bias result: Fig-1A "PLS>Ridge" is partly best-k-per-draw inflation; honest fixed-k → PLS>Ridge only for qwen3_32b.
- [x] **TF-IDF 4th curve** — cluster sbatch timed out (run_mc_probes tfidf path does 6× redundant work); computed lean locally via `tfidf_ksweep_local.py` (~13 min, parallel). tfidf peaks k=3 (0.376) **above its Ridge** (0.355). Refined rule: **PLS>Ridge exactly for the high-dim spaces (qwen3_32b 5120-dim, tfidf ~8000-dim sparse); Ridge≥PLS for low-dim mlm/thalesian.**
- Figs (all 4 models now): `pls_components_tradeoff`, `per_method_panels`, `fixed_k3_pls_vs_ridge`, `best_k_vs_fixed_k`.

## Task 3 — error analysis  (`error_overlap/`)
The 2×2 design = {imbalanced, balanced} × {tier0, maximal}:

| | tier0 (full length) | maximal (≤32 words) |
|---|---|---|
| **imbalanced** | [x] `predictions/` (4 models) | [x] `predictions_maximal/` (3 models) |
| **balanced** | [x] `predictions_tier0_balanced/` (job 9526) | [x] `predictions_maximal_balanced/` (job 9437) |

- [x] Length is main axis; qwen3_32b short-text specialist; thalesian length-hungry.
- [x] Universal failures = rare-period + very-short (+ scribal damage `he-pi2`/`eššu`).
- [x] No model has unique dating power (unique-wins≈0; JS 0.08–0.11) → one shared signal.
- [x] Maximal length-control: equalizing length collapses TF-IDF's lead, halves length slopes.
- [x] Balanced+maximal (both crutches off): thalesian Sp 0.41 leads; acc@100 is meaningless (188-yr span, dummy=0.937) → use **±25 vs dummy floor** + MAE/std + Spearman; signal weak at fine resolution.
- [x] **Balanced+tier0 (4th cell, job 9526)** — 2×2 complete. **Result: with classes balanced, the rank flips by length.** tier0 (full text): tfidf 0.474 > thalesian 0.430 > qwen 0.311. maximal (≤32w): thalesian 0.407 > qwen 0.276 > tfidf 0.245. **TF-IDF's lead is purely length** (Δ +0.228); thalesian/qwen length-robust (Δ +0.02–0.04). So thalesian "winning" balanced-maximal = it's just the last model standing when text is truncated, NOT a class-balancing edge. Fig: `predictions_tier0_balanced/balanced_length_compare.png` (script `balanced_length_compare.py`).

### Analysis scripts (`error_overlap/`)
`dump_oof_predictions.py` (--cleaning), `dump_oof_predictions_balanced.py` (--cleaning, per-draw best-k),
`error_map.py`, `analyze_per_model.py`, `compare_anchor.py` (--anchor tfidf), `tier0_vs_maximal.py`,
`balanced_length_compare.py` (balanced tier0-vs-maximal), `balanced_summary.py` (scale-aware
summary + accuracy@N + chronology for any predictions_* dir; wired into both balanced sbatchs).

## Workflow reminders
- Yarin runs ONLY `sbatch` on cluster (no interactive python); activations cluster-only.
- Cluster sbatch commits+pushes results → `git pull` locally. `yarin/` is gitignored.
- conda env `thesis`; cluster repo `~/projects/HUJI-THESIS--YARIN`.

## Open metric caveat (don't forget)
On the balanced sets the year span is tiny (8 rulers ≈ 539–727 BCE = 188 yr), so
acc@±100 is trivially ~0.94 from a predict-the-mean dummy. **Always report ±25 yr
vs the dummy floor, plus MAE/std and Spearman.**
