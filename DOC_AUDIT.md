# Documentation Audit
**Date:** 2026-05-17 · **Auditor:** Claude (Opus 4.7) · **Owner:** Yarin Benizri

Audit of every `.md` / `.txt` in the repo (excluding `papers/`, `venv/`, `.git/`, `node_modules/`, `__pycache__/`, `.claude/agents/`). Focus: orientation, staleness, upcoming directory rename (`orcc_round1/`, `seal_round4/` under `src/linear_probing/results/`).

---

## 1. Orientation reading order

For a fresh Claude session or new collaborator. 5 files, ~30 min:

1. **`/README.md`** — One-screen project pitch + phase status table + repo map. Anchors mental model.
2. **`/v_1/PROGRESS.md`** — Most recent narrative status snapshot (dated 2026-04-14). Tells you what is done / blocked / next.
3. **`/v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md`** — The single source of truth for Track B experimental results (Steps 00–08, headline findings, exact numbers, scripts).
4. **`/PLAN_round2_qwen_diagnosis.md`** — The *current* active research plan (Round 2, dated 2026-05-15). Tells you where the project is heading next; supersedes the older "next steps" lists.
5. **`/RESEARCH_LOG.md`** — Condensed milestone history (Phases 0–1.5 + Track A). Useful for the thesis writeup; less useful for "what am I doing today."
6. *(optional)* **`/v_1/src/viz/README.md`** — Best entry point if your immediate task is the embedding explorer GUI.
7. *(optional)* **`/v_1/src/sae/plan/PLAN.md`** — Only if working Track C / SAE.

Skip on first pass: `justification/` (decision archive, only relevant when writing thesis), `PLAN_viz_extension.md` (marked complete 2026-04-28), `yarin/` (personal planning, not project state).

---

## 2. File map

Sorted by importance/freshness (most current+load-bearing first).

| Path | Purpose | Audience | Last-updated | Status |
|---|---|---|---|---|
| `/v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md` | Single-source-of-truth run log for Track B (Steps 00–08, all numbers, scripts, headline findings) | Yarin / Future collaborators | Step 08 dated 2026-05-11 | ⚠ paths reference `orcc_round1/` heavily (post-rename stale) |
| `/PLAN_round2_qwen_diagnosis.md` | Active research plan: 3-phase Qwen-failure diagnosis | Yarin / Phase 1–3 subagents | 2026-05-15 | ✅ current (just-authored) |
| `/v_1/PROGRESS.md` | Mid-level status snapshot: tracks, bias check, SEAL pipeline, next steps | Yarin / Future collaborators | Header date 2026-04-14 | ⚠ stale — predates ORCC/PLS work (Steps 05–08 of run log); "Next Steps" item 4 is done |
| `/README.md` | Repo-root pitch + phase table + folder map | Future collaborators | "Mar 2026" header | ⚠ phase table claims Track B "IN PROGRESS" but PIPELINE_RUN_LOG shows letters+ORCC complete |
| `/v_1/README.md` | v_1 overview: folder structure, Track A/B quickstarts, key files, SEAL pipeline status | Yarin / Future collaborators | "as of 2026-04-26" | ⚠ paths in lines 83-84 will go stale; line 88-89 references results files at paths that will move |
| `/v_1/src/viz/README.md` | Embedding explorer GUI: controls, data layout, file map | Yarin / Future collaborators | "v4, 2026-05-11" | ⚠ lines 185, 192, 202, 212, 250-254 reference `seal_round4/`+`orcc_round1/` result paths |
| `/RESEARCH_LOG.md` | Condensed milestone+results history Phase 0 → Track A | Yarin (thesis writeup) | "Apr–May 2026" pending | ⚠ stops at Track A; doesn't mention Track B/PLS/ORCC; line 112 dead link `v_1/data/evaluation_corpora/` (should be `evaluation/corpora/`) |
| `/v_1/src/sae/plan/PLAN.md` | Track C SAE implementation plan (Arditi 131k, layers 7/15/23) | Yarin / SAE subagent | "April 2026" | ✅ current; status tracker (lines 530-542) shows nothing done yet — but git status shows fresh `v_1/src/sae/` untracked work |
| `/justification/seal_round4_pipeline_plan.md` | The big SEAL pipeline design doc (773 lines, 19 sections) | Yarin (thesis) / SEAL subagent | "Phase C complete 2026-04-07" | ⚠ sections 16–19 still valid as the "verified facts" reference but plan-of-action prose is historical |
| `/PLAN_viz_extension.md` | 5-feature viz extension plan with 3 terminal scripts | (Historical) | "COMPLETE 2026-04-28" | 🗄 archive candidate — work is done, references future-tense |
| `/justification/parallel_plans_final.md` | Multi-plan execution doc for SEAL round-5 + EDA GUI | (Historical) | "last updated 2026-04-14" | 🗄 archive candidate — plans A–E all marked ✅ Done; superseded by PROGRESS.md "Parallel Execution Status" |
| `/v_1/src/linear_probing/results/orcc_round1/pls/CV_STRATEGY_NOTE.md` | PLS CV-strategy methodology note (GroupKFold-by-ruler tradeoffs) | Yarin (thesis methods) / Future collaborators | 2026-05-05 | ⚠ pipeline version "orcc_round1 jobs 5561–5563" outdated (current jobs per run log are 6522/6523/6433); will need to move when results dir renamed |
| `/justification/chunrong_data_cleaning_decisions.md` | Documented data-cleaning decisions w/ Chunrong (4-round email log) | Yarin (thesis) | "March 8, 2026" | ✅ current as a historical record |
| `/justification/evaluation_corpus_size_5005_texts.md` | Corpus-size decision (5005 → 4976 → 4957 after cleanup) | Yarin (thesis) | "January 28, 2026" | ✅ current as historical record; filename misleading (we landed at 4957) |
| `/justification/domain_label_cleanup.md` | Why 29 Unknown-domain texts were dropped (5005 → 4976) | Yarin (thesis) | "January 28, 2026" | ✅ current historical record |
| `/justification/cdli_oracc_metadata_matching.md` | How ORACC period/genre metadata was joined to CDLI | Yarin (thesis) | "January 2026" | ✅ current historical record |
| `/justification/justification_mlm.md` | Why MLM (BERT-style) over causal LM | Yarin (thesis) | "December 2025" | ⚠ Phase 1 MLM marked "superseded" in README; doc itself is still valid rationale |
| `/justification/justification_sign_level_tokenization.md` | Why sign-level tokenization | Yarin (thesis) | "December 2025" | ⚠ same — design rationale for an architecture no longer central |
| `/justification/justification_aeneas_twin_architecture.md` | Why the 16-layer Simplified Aeneas architecture | Yarin (thesis) | "December 2025" | 🗄 archive candidate — model superseded by fine-tuning pretrained LLMs |
| `/justification/VALIDATION_PHASE1_TRAINING.md` | Phase 1 MLM training validation (476 lines) | Yarin (thesis) | "December 29, 2025" | 🗄 archive candidate — Phase 1 superseded; preserve for thesis methods chapter |
| `/justification/justification_data_pipeline_verification.md` | Data pipeline verification trace | Yarin (thesis) | "January 2026" | ✅ current historical record |
| `/justification/data_source_summary.md` | Unified dataset composition (eBL/ORACC/Archibab) | Yarin (thesis) | "December 2025" | ✅ current historical record |
| `/justification/model_selection_phase2.md` | LLM tiers chosen for Track A (Jan 2026) | Yarin (thesis) | "January 28, 2026" | ⚠ partially stale — Phase 2 plan; check vs current Track A status |
| `/justification/task_2_5_metadata_audit_summary.md` | Metadata audit deliverable | Yarin (thesis) | "January 2026, ✅ COMPLETED" | ✅ historical record |
| `/v_1/data/README.md` | Data-directory overview (pipeline flow + folder summary) | Future collaborators | undated | ✅ current |
| `/v_1/data/evaluation/corpora/README.md` | Eval corpora files including `seal_corpus.parquet` + `orcc_corpus.parquet` | Yarin / Future collaborators | "2026-04-26" | ✅ current |
| `/v_1/data/evaluation/README.md` | Evaluation root | Future collaborators | undated | (not read in full) |
| `/v_1/data/raw/chungrong/README.md` | Letter CSVs (archibab/oracc/lbl) | Future collaborators | undated | ✅ current |
| `/v_1/data/raw/chungrong/seal_round4/README.md` | SEAL/DLL/LBPL round-4 delivery (raw data, NOT renaming) | Future collaborators | "April 2026" | ✅ current — this `seal_round4` is a raw-data folder name; OUT OF SCOPE for rename |
| `/v_1/data/raw/chungrong/seal_round4/inspection_report.md` | Phase 0 inspection output | Yarin | "Apr 4-6, 2026" | ✅ current |
| `/v_1/data/evaluation/bias_check/seal_round4/README.md` | Phase C re-run results (round-5 data) | Yarin (thesis) | "2026-04-14" | ✅ current — bias_check is co-located with raw delivery name; OUT OF SCOPE for rename |
| `/v_1/data/evaluation/bias_check/seal_round4/label_issues.md` | Confusion-matrix label-quality report | Yarin / Chunrong | "2026-04-09" | ✅ current |
| `/v_1/data/evaluation/corpora/seal_tasks_verification.md` | Task registry self-test report (Phase B) | Yarin | "Phase B" 2026-04-07 | ✅ current |
| `/v_1/data/evaluation/baselines/README.md` | Baselines folder | Future collaborators | undated | (not deeply audited) |
| `/v_1/data/evaluation/baselines/baseline_results_report.md` | Track A baseline result report | Yarin (thesis) | undated | (not deeply audited) — likely ✅ |
| `/v_1/src/evaluation/README.md` | Track A LLM baseline pipeline quickstart | Future collaborators | undated | ✅ current |
| `/v_1/src/bias_check/README.md` | Bias check pipeline (full + SEAL multi-task variant) | Future collaborators | undated | ⚠ line 229 references `data/evaluation/bias_check/seal_round4/` (raw-data path, OK to keep) |
| `/v_1/src/cluster/README.md` | Schmidt HPC setup, conda env, slurm | Future collaborators | undated | (not deeply audited) |
| `/v_1/data/external/README.md`, `processed/README.md`, `training_ready/README.md`, `unified/README.md`, `analysis_outputs/README.md` | Subdirectory READMEs | Future collaborators | undated | (not deeply audited) — likely ✅ as static folder docs |
| `/v_1/data/evaluation/bias_check/seal_round4/{period,genre,sub_genre,domain,provenance,sub_provenance}/{tier0,maximal}/report.md` (×12) | Generated bias-check per-task reports | Yarin | regenerated 2026-04-14 | ✅ current (auto-generated artifacts) |
| `/v_1/data/evaluation/bias_check/bias_check_report.md`, `bias_check_summary.md` | Letters bias-check reports | Yarin | 2026-03 | ✅ current (auto-generated artifacts) |
| `/yarin/research_plan.md` | Yarin's personal research plan | Yarin | "2026-03-15" | ⚠ pre-Track B; mentions evaluation corpus as 4,976; not part of repo orientation |
| `/yarin/research_plan/README.md`, `planning/stream1_taboola.md`, `planning/stream2_amir_connection.md`, `work/work_situatoin.txt` | Personal career/research planning | Yarin only | varies | OUT OF SCOPE (personal) — exclude from project audit |
| `/yarin/meeting_notes.md` | Personal meeting notes | Yarin only | varies | OUT OF SCOPE (personal) |
| `/yarin/emails_phase/round{1,2,3,4}/...` + `archive/project_origins.txt` | Email log + project origin notes | Yarin only | historical | ✅ historical archive — leave alone |
| `/yarin/emails_phase/round1/EMAIL_ANALYSIS_AND_DATA_VERIFICATION.md` | One-off analysis | Yarin | historical | 🗄 historical |
| `/deep-research-report (4).md` | LLM "deep research" report on existing Akkadian HF models | Yarin (background) | undated (new, untracked) | ⚠ informational; not integrated anywhere; consider moving to `papers/` or `justification/background/` |
| `/PLAN_viz_extension.md` | Viz-extension plan (5 features, 3 terminals) | (Historical) | "COMPLETE 2026-04-28" | 🗄 ARCHIVE — done |
| `/v_1/data/unified/dataset_stats.txt`, `eda_summary.txt` | Generated stat dumps | Yarin | undated | ✅ generated artifacts |
| `/v_1/src/cluster/setup_emails.txt` | Cluster setup email log | Yarin | historical | ✅ historical |
| `/v_1/data/raw/downloaded_data_explained.txt` | Raw-data manifest | Yarin / collaborators | historical | ✅ current |
| `/v_1/requirements.txt`, `/yarin/requirements.txt` | Python deps | Future collaborators | undated | ✅ current (not docs but in scope) |

**Note on `.claude/agents/`** — not audited per instructions; PLAN_round2_qwen_diagnosis.md references `phase1-eliciter.md`, `phase2-scaler-sae.md`, `phase3-tokenizer.md`, `prompt-drafter.md`, `eval-harness-builder.md`, `cluster-job-runner.md`, `phase-synthesizer.md`. Verify these exist before Phase 1 launch.

---

## 3. Staleness & mismatches

### 3a. Will go stale after the planned directory rename
Every line below references a `src/linear_probing/results/{orcc_round1,seal_round4}/...` path that is about to be renamed. These need updates after the rename runs.

- `v_1/PROGRESS.md:64` — `data/evaluation/bias_check/seal_round4/...` — this is the BIAS_CHECK path (data/evaluation), which is OUT OF SCOPE for the rename. Keep.
- `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md:750` `results/orcc_round1/pls/` (Step 05 output schema)
- `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md:787` `results/orcc_round1/cls/` (Step 05b)
- `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md:821` `results/orcc_round1/pls/figures/`
- `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md:834` `results/orcc_round1/cls/figures/`
- `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md:893` block diagram `└── results/orcc_round1/` (lines ~893–905)
- `v_1/src/linear_probing/results/orcc_round1/pls/CV_STRATEGY_NOTE.md:4` `Pipeline version: orcc_round1 PLS jobs 5561–5563`
- `v_1/src/viz/README.md:185` `seal_round4/seal_qwen_coords.json`
- `v_1/src/viz/README.md:192` `seal_round4/seal_mlm_coords.json`
- `v_1/src/viz/README.md:202` `seal_round4/seal_qwen_coords_last.json`
- `v_1/src/viz/README.md:212` `orcc_round1/orcc_qwen_coords_mean.json`, `orcc_qwen_coords_last.json`
- `v_1/src/viz/README.md:250` `src/linear_probing/results/seal_round4/seal_qwen_coords.json`
- `v_1/src/viz/README.md:251` `src/linear_probing/results/seal_round4/seal_mlm_coords.json`
- `v_1/src/viz/README.md:252` `src/linear_probing/results/seal_round4/seal_qwen_coords_last.json`
- `v_1/src/viz/README.md:253` `src/linear_probing/results/orcc_round1/orcc_qwen_coords_mean.json`
- `v_1/src/viz/README.md:254` `src/linear_probing/results/orcc_round1/orcc_qwen_coords_last.json`
- `v_1/README.md:75` `src/linear_probing/results/PIPELINE_RUN_LOG.md` — path itself fine; OK
- `PLAN_viz_extension.md:118–138, 376, 393, 395, 422, 425, 437, 439, 444, 446, 457–460, 505, 510` — many `seal_round4/`+`orcc_round1/` paths under `src/linear_probing/results/`. Lower priority since this doc is archive-candidate.
- `justification/parallel_plans_final.md:274, 282, 302, 313, 322, 323, 400` — references `results/seal_round4/activations/...`. Lower priority since archive-candidate.

**Confirmed NOT stale (raw-data folder names — out of rename scope):**
- All `v_1/data/raw/chungrong/seal_round4/...` paths
- All `v_1/data/evaluation/bias_check/seal_round4/...` paths
- All `v_1/data/raw/chungrong/orcc_round1/...` paths

### 3b. "Not yet started / pending" that is actually done
- `README.md:14` — phase table says `Track A: IN PROGRESS`. Cross-check with `RESEARCH_LOG.md:14` which says "Phase 2 / Track A: IN PROGRESS Jan–Mar 2026" — likely fine, but `PROGRESS.md` doesn't update the Track-A status table. Reconcile.
- `README.md:14` says `Track B: PENDING`. **Contradicts** `PROGRESS.md:11`: "Track B: Linear probing of Qwen2.5-7B internal representations — complete on letters corpus" AND PIPELINE_RUN_LOG Steps 01–08 (all ✅). Bring `README.md` in sync.
- `README.md:14` says `Track C: PENDING`. `PROGRESS.md:13` says "Track C: SAE analysis — planned". `v_1/src/sae/PLAN.md` status tracker shows nothing started but git status shows untracked `v_1/src/sae/` work. Reconcile actual state.
- `PROGRESS.md:144` "Next Steps" item 4 says write+run `03_extract_seal_embeddings.py` — but Plan D-extraction is marked ✅ Done elsewhere in the same doc (`PROGRESS.md:120`, `:131`). Internal contradiction.
- `v_1/src/sae/plan/PLAN.md:530-542` status tracker — all unchecked. Verify against untracked `v_1/src/sae/` files in git status before claiming nothing is done.

### 3c. Two docs giving contradictory facts
- **Eval corpus size:** `yarin/research_plan.md:10` says **4,976 texts**. `README.md:11`, `v_1/PROGRESS.md`, `RESEARCH_LOG.md:112`, `v_1/README.md:81` all say **4,957 texts**. `justification/domain_label_cleanup.md` (line 4) says **4,976 (down from 5,005)** but `justification/evaluation_corpus_size_5005_texts.md` filename says **5,005**. The current canonical number is **4,957** (per PROGRESS.md + PIPELINE_RUN_LOG line 5 + bias check). The 4,976→4,957 delta is unexplained in docs but the number 4,957 is consistent across run artifacts. → Fix `yarin/research_plan.md:10`; consider renaming `justification/evaluation_corpus_size_5005_texts.md` → `evaluation_corpus_size_4957_texts.md`.
- **Best layer (Qwen letters, mean pooling, tier0):** `PROGRESS.md:24` says `L4`. PIPELINE_RUN_LOG line 161 says `L4`. ✅ consistent.
- **Best layer (Qwen letters, mean pooling, maximal):** `PROGRESS.md:25` says `L3`. PIPELINE_RUN_LOG line 169 says `L3`. ✅ consistent.
- **MLM val_loss:** `v_1/README.md:108` says `2.9777`. `v_1/PROGRESS.md:118` says `2.9777`. `RESEARCH_LOG.md:79` says `3.0204` (this is the OLD Phase-1 MLM, different model). Mild risk of confusion; clarify which MLM you mean.
- **Headline finding ranking:** `PIPELINE_RUN_LOG.md:857` says "TF-IDF >> MLM ≈ Random > Qwen" (ORCC, year+ruler). `PIPELINE_RUN_LOG.md:473` (letters Step 02) shows Qwen *beats* TF-IDF for mean-pooling. **Both are correct** but appear contradictory if read out of context. ORCC vs letters is the key disambiguator. Currently the only place this is reconciled is via reading both sections.

### 3d. Phase/step numbering inconsistencies
- `PIPELINE_RUN_LOG.md` jumps from Step 02b directly to Step 05 (no Step 03 or 04 in current log). Step 03 is mentioned as "Not yet run" at line 689. Step 04 is missing entirely. Fine but confusing.
- `seal_round4_pipeline_plan.md` Section numbers are referenced as 16, 17, 18, 19 — verify these exist in the 773-line file (only first 100 lines audited).
- `justification/parallel_plans_final.md:60` references `v_1/justification/seal_round4_pipeline_plan.md` — but the file is at `/justification/seal_round4_pipeline_plan.md` (no `v_1/` prefix). **Dead link.** Same dead `v_1/justification/...` reference at lines 115, 255, 349.

### 3e. Dead relative links
- `RESEARCH_LOG.md:112` `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet` — correct path is `v_1/data/evaluation/corpora/...` (no underscore between `evaluation` and `corpora`).
- `justification/parallel_plans_final.md:60, 115, 255, 349` — `v_1/justification/seal_round4_pipeline_plan.md` does not exist; actual path is `justification/seal_round4_pipeline_plan.md`.
- `README.md:43` `RESEARCH_LOG.md` — exists ✅.
- `v_1/README.md:124` `justification/seal_round4_pipeline_plan.md` — written as relative to v_1; actual file is at repo root `/justification/seal_round4_pipeline_plan.md`. From v_1 the correct relative is `../justification/seal_round4_pipeline_plan.md`. ⚠ broken-when-rendered.
- `v_1/PROGRESS.md:41` `justification/seal_round4_pipeline_plan.md` — same issue (relative from v_1/).
- `PLAN_round2_qwen_diagnosis.md:67, 156-165` references `v_1/src/embeddings/` and `.claude/agents/*.md` — verify these exist before Phase 1 execution. (Quick check: `v_1/src/embeddings/` was not in the audit file list — possibly doesn't exist yet.)

### 3f. Per-doc internal inconsistencies
- `v_1/PROGRESS.md` header says "Status Date: 2026-04-14" but body mentions cluster jobs through 2026-04-15 (line 131). Consider bumping date when next edited.
- `PROGRESS.md:139` lists blockers (advisor full dataset, fine-grained labels) that are not echoed in `PLAN_round2_qwen_diagnosis.md` — Round 2 takes a different angle (Qwen diagnostics, no new data required). Document the pivot.

---

## 4. Reorganization proposal

### Archive (move to `/archive/` at repo root)
| File | Reason |
|---|---|
| `PLAN_viz_extension.md` | Marked COMPLETE 2026-04-28; viz GUI is live |
| `justification/parallel_plans_final.md` | All five plans marked ✅ Done; superseded by PROGRESS.md "Parallel Execution Status" table |
| `justification/VALIDATION_PHASE1_TRAINING.md` | Phase 1 MLM superseded by fine-tuning approach (per README.md:11) |
| `justification/justification_aeneas_twin_architecture.md` | Architecture no longer used; keep for thesis methods chapter but archive from active dir |
| `justification/task_2_5_metadata_audit_summary.md` | One-off January 2026 deliverable; ✅ COMPLETED |
| `yarin/emails_phase/round1/EMAIL_ANALYSIS_AND_DATA_VERIFICATION.md` | Historical |

Suggested path: `/archive/2026-q1/` (group by quarter so future archives are easy).

### Merge
| Merge | Into | Reason |
|---|---|---|
| `README.md` (root) + `v_1/README.md` | Keep both, but rewrite root README as a 1-screen pointer that redirects to v_1/README.md and PROGRESS.md | Currently both contain folder maps; redundant and the root one is staler (says Track B "PENDING") |
| `justification/evaluation_corpus_size_5005_texts.md` + `justification/domain_label_cleanup.md` | Single doc `justification/evaluation_corpus_construction.md` walking 5005 → 4976 → 4957 | They tell consecutive chapters of the same story |
| `justification/justification_mlm.md` (45 lines) + `justification/justification_sign_level_tokenization.md` (49 lines) + `justification/justification_aeneas_twin_architecture.md` | `justification/phase1_mlm_design.md` (archived) | All cover the now-superseded Phase 1 MLM design |

### Split
| Doc | Split into |
|---|---|
| `PIPELINE_RUN_LOG.md` (937 lines) | Keep one log, but consider splitting Step 00–04 (letters) and Step 05–08 (ORCC) into two appendix files; current single file is becoming hard to navigate. Or add a generated TOC at top. |
| `justification/seal_round4_pipeline_plan.md` (773 lines) | Keep "Sections 16–19 verified facts" as standalone reference doc `seal_corpus_facts.md`; archive the prose plan-of-action |

### Rename
| Old | New | Reason |
|---|---|---|
| `justification/evaluation_corpus_size_5005_texts.md` | `evaluation_corpus_size_4957_texts.md` (or merged per above) | Filename has wrong number |
| `deep-research-report (4).md` | `justification/background/akkadian_model_landscape_2026-05.md` (or move to `papers/`) | Has a space and meaningless `(4)`; should live with research notes |
| `v_1/src/linear_probing/results/orcc_round1/{cls,pls,coord JSONs}` | (already planned: `orcc__probe_cls/`, `orcc__probe_pls/`, `orcc__embed/`) | Per user's note |
| `v_1/src/linear_probing/results/seal_round4/` | (already planned: `seal__embed/`) | Per user's note |
| Letters root-level result files in `v_1/src/linear_probing/results/` | (already planned: `letters__probe_cls__period/`) | Per user's note |
| `yarin/research_plan/work/work_situatoin.txt` | `work_situation.txt` | Typo |

---

## 5. Cross-reference plan

Proposed "See also" header for top of each major doc. Insert immediately after H1 title.

**`/README.md`** (root)
```
> **See also:** [v_1/README.md](v_1/README.md) for working dir layout · [v_1/PROGRESS.md](v_1/PROGRESS.md) for current status · [PLAN_round2_qwen_diagnosis.md](PLAN_round2_qwen_diagnosis.md) for active research plan
```

**`/v_1/README.md`**
```
> **See also:** [PROGRESS.md](PROGRESS.md) for current status · [src/linear_probing/results/PIPELINE_RUN_LOG.md](src/linear_probing/results/PIPELINE_RUN_LOG.md) for Track B results · [../RESEARCH_LOG.md](../RESEARCH_LOG.md) for thesis-writing milestone log
```

**`/v_1/PROGRESS.md`**
```
> **See also:** [README.md](README.md) for repo orientation · [src/linear_probing/results/PIPELINE_RUN_LOG.md](src/linear_probing/results/PIPELINE_RUN_LOG.md) for all probe results · [../PLAN_round2_qwen_diagnosis.md](../PLAN_round2_qwen_diagnosis.md) for next-phase plan
```

**`/RESEARCH_LOG.md`**
```
> **See also:** [v_1/PROGRESS.md](v_1/PROGRESS.md) for current status · [v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md](v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md) for Track B/PLS results · [justification/](justification/) for decision docs
```

**`/v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md`**
```
> **See also:** [../../../PROGRESS.md](../../../PROGRESS.md) for project status · [../../../../PLAN_round2_qwen_diagnosis.md](../../../../PLAN_round2_qwen_diagnosis.md) for the Round 2 follow-up plan based on these results
```

**`/PLAN_round2_qwen_diagnosis.md`**
```
> **See also:** [v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md](v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md) Steps 05–08 for the Round 1 findings this plan reacts to · [v_1/src/sae/plan/PLAN.md](v_1/src/sae/plan/PLAN.md) for Track C SAE plan (Phase 2 dependency)
```

**`/v_1/src/sae/plan/PLAN.md`**
```
> **See also:** [../../linear_probing/results/PIPELINE_RUN_LOG.md](../../linear_probing/results/PIPELINE_RUN_LOG.md) for the linear probe results this builds on · [../../../../PLAN_round2_qwen_diagnosis.md](../../../../PLAN_round2_qwen_diagnosis.md) Phase 2 for Qwen-3 SAE follow-up
```

**`/v_1/src/viz/README.md`**
```
> **See also:** [../linear_probing/results/PIPELINE_RUN_LOG.md](../linear_probing/results/PIPELINE_RUN_LOG.md) Step 08 for the PLS reductions exposed in the GUI · [../../PROGRESS.md](../../PROGRESS.md) for overall project status
```

---

## 6. Post-rename update list

After the planned rename (`orcc_round1/cls → orcc__probe_cls/`, `orcc_round1/pls → orcc__probe_pls/`, `orcc_round1/{coord JSONs} → orcc__embed/`, `seal_round4/ → seal__embed/`, root letters files → `letters__probe_cls__period/`), update these:

### High-priority (still in active use)
1. **`v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md`** — sections:
   - Step 05 "Outputs (per method)" block at lines ~749–763 (the `results/orcc_round1/pls/` code block)
   - Step 05b "Outputs (already on cluster, pre-existing)" block at lines ~786–792 (`results/orcc_round1/cls/`)
   - Step 06 "**`pls_best_layers.json`**" / "**`cls_best_layers.json`**" prose, lines ~801–812 (mentions paths implicitly)
   - Step 07 "in `results/orcc_round1/pls/figures/`" line 821 and `results/orcc_round1/cls/figures/` line 834
   - "File Map (Step 05 onwards)" code block lines ~876–911 (most concentrated path block)
   - "How to Reproduce From Scratch" block lines ~915–936 if any path mentioned (mostly script paths, not result paths — verify)
2. **`v_1/src/viz/README.md`** — lines 185, 192, 202, 212, 250–254 (Cluster Output, Local Files table). Update file paths and recompute byte sizes if needed.
3. **`v_1/src/linear_probing/results/orcc_round1/pls/CV_STRATEGY_NOTE.md`** — this file *itself* will move to `orcc__probe_pls/CV_STRATEGY_NOTE.md`; update line 4 `**Pipeline version:** orcc_round1 PLS jobs 5561–5563` (also: those jobs are stale — current per run log are 6522/6523).
4. **`v_1/README.md`** — line 88-89 (Key Data Files table) reference `src/linear_probing/results/probe_results_qwen2.5-7b-instruct.json` and `validity_results_*.json` — verify these letters-pipeline files move to `letters__probe_cls__period/` and update.
5. **`v_1/PROGRESS.md`** — section "Track B — Linear Probing" line 37 references `src/linear_probing/results/PIPELINE_RUN_LOG.md` (path itself unchanged; OK).

### Lower-priority (archive candidates)
6. **`PLAN_viz_extension.md`** — many `seal_round4/`+`orcc_round1/` paths (lines 118–138, 376, 393, 395, 422–460, 505, 510). If archived, just add a header note that paths reference pre-rename layout.
7. **`justification/parallel_plans_final.md`** — lines 274, 282, 302, 313, 322, 323, 400. Same note if archived.

### No update needed (data folders, NOT renaming)
- All `v_1/data/raw/chungrong/seal_round4/*` references
- All `v_1/data/evaluation/bias_check/seal_round4/*` references
- All `v_1/data/raw/chungrong/orcc_round1/*` references
- `v_1/src/bias_check/README.md:229`

---

## 7. Quick wins

Five low-effort, high-impact edits:

1. **Add a `## TL;DR` block to top of `/README.md`** (5 lines): current status (Track A+B done on letters+ORCC, Round 2 Qwen-diagnosis active), where to read next (PROGRESS → PIPELINE_RUN_LOG → PLAN_round2). Will save every new collaborator ~10 minutes.

2. **Fix the README phase table (`/README.md:7–15`)**: Track B is shown as PENDING but is complete on both letters and ORCC. Update to: `Track A: COMPLETE on letters · IN PROGRESS for full model sweep`, `Track B: COMPLETE on letters and ORCC (Round 1) · ROUND 2 IN PROGRESS`, `Track C: SAE planning + initial extraction (see git untracked)`.

3. **Stamp every top-level doc with a "Last verified: YYYY-MM-DD" line** so freshness can be assessed at a glance. Especially: `README.md`, `v_1/README.md`, `RESEARCH_LOG.md`, `PROGRESS.md`. Use `**Last verified:** 2026-05-17 (Yarin)` pattern.

4. **Fix the four dead `v_1/justification/...` links** in `justification/parallel_plans_final.md:60, 115, 255, 349` and the eval-corpora path in `RESEARCH_LOG.md:112`. Total edit: 5 lines across 2 files. Trivial but blocks any reader's flow.

5. **Add memory note to `/Users/yarin.b/.claude-personal/projects/-Users-yarin-b-git-lititure-review/memory/`** capturing two facts repeatedly needed across sessions: (a) `seal_round4`/`orcc_round1` are reused as BOTH raw-data folder names AND results-folder names — only the latter is being renamed; (b) the canonical letters corpus size is **4,957** (not 4,976 or 5,005). These keep getting re-derived from scratch.

*(Bonus low-effort win)*: Add a one-line "What is this?" to the orphan `/deep-research-report (4).md` — at minimum a date and "ad-hoc LLM survey of Akkadian pretrained checkpoints; informational only, not part of pipeline."
