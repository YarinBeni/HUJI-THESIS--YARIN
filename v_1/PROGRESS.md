# Project Progress & Handover Snapshot

> **Status Date:** 2026-04-07
> **Current Phase:** SEAL pipeline Phase C complete. Phase D (linear probing on cluster) is next. Track C (SAE) plan finalized, ready after SEAL.
> **Working Directory:** `v_1/`

## Project Context

Three-track thesis on mechanistic interpretability of LLMs applied to Akkadian cuneiform temporal dating:

- **Track A:** LLM baseline evaluation (OpenRouter API) — complete
- **Track B:** Linear probing of Qwen2.5-7B internal representations — complete on letters corpus
- **Track C:** Sparse Autoencoder (SAE) analysis — planned, awaiting full dataset

## Current Status (March 2026)

### Bias Check: Complete
TF-IDF classification on 4,957 letters: 98.3% (tier0), 91% (maximal cleaning). Signal is genuine diachronic linguistic change, not dataset bias. Full report: `data/evaluation/bias_check/`.

### Track B — Linear Probing: Complete (letters corpus)
Full pipeline on Qwen2.5-7B-Instruct, 4,957 letters, 29 layers, mean + last_token pooling, tier0 + maximal cleaning.

**Key results:**
| Condition | Pretrained | Random | Gap |
|-----------|-----------|--------|-----|
| Mean, tier0 (L4) | 99.1% | 98.3% | +0.8% |
| Mean, maximal (L3) | 96.3% | 90.7% | +5.5% |
| Last_token, tier0 (L28) | 95.5% | 84.8% | +10.7% |
| Last_token, maximal (L28) | 90.0% | 70.1% | +19.9% |

**Validity tests (all complete):**
- Learning curve: mean pooling hits 93% with just 42 texts (1%)
- PCA: top-5 PCs recover 90% accuracy (mean pooling) — very compact signal
- MLP vs linear: MLP < linear everywhere — genuinely linear encoding
- Random baseline: +20% pretraining gap for last_token/maximal (strongest selectivity)

Full log: `src/linear_probing/results/PIPELINE_RUN_LOG.md`

### SEAL / DLL / LBPL Pipeline: Phase B Complete (2026-04-07)

Full plan: `justification/seal_round4_pipeline_plan.md` (Sections 16–17 have all verified facts).

**Phase 0 — Inspection** ✅
- 384 fragments / 40,484 words across SEAL+DLL+LBPL profiled by `src/corpus/01_inspect_seal_data.py`.
- Report: `data/raw/chungrong/seal_round4/inspection_report.{md,json}`.
- Known issue (not a blocker): 4 rows in `dll.csv` have null `clean_value` (digit-only tokens). Fallback to `value` pinned to exactly these 4 rows.

**Phase A — Corpus build** ✅
- `src/corpus/02_build_seal_corpus.py` built `data/evaluation/corpora/seal_corpus.parquet` (384 rows × 15 cols).
- Columns: `fragment_id, corpus, word_language, domain, period, genre, sub_genre, provenance, sub_provenance, place_discovery, place_composition, word_count, text, text_tier0, text_maximal`
- `genre`/`sub_genre` are lowercase+stripped. `text_tier0`/`text_maximal` pre-computed.
- All MD5, fallback, and spot-check assertions passed.
- Summary: `data/evaluation/corpora/seal_corpus_summary.json`.

**Phase B — Task registry** ✅
- `src/bias_check/seal_tasks.py` — `load_task_data(task_name, cleaning)` returns `(df, task_summary)`.
- 6 tasks verified against inspection contract: period/genre/provenance/sub_provenance/domain all use SEAL+DLL+LBPL; sub_genre uses SEAL only.
- All 6 tasks ✓ PASS self-test. Verification: `data/evaluation/corpora/seal_tasks_verification.md`.
- Only `domain` reaches k=5; all other tasks k=2 (smallest surviving class has N=2).

**Phase C — Bias check CV** ✅ (re-run 2026-04-14 on round 5 data)
- Script: `src/bias_check/06_bias_check_cv.py` — TF-IDF char_wb(2,5) + LR + adaptive-k stratified CV + 1000-perm test
- All 12 combinations (6 tasks × 2 cleanings) re-ran locally on round 5 CSVs. All FAIL (p=0.001).
- Per-task outputs: `data/evaluation/bias_check/seal_round4/<task>/<cleaning>/` (metrics.json, report.md, plots/)
- Key results (round 5): domain F1=0.952 (tier0), period F1=0.473 (9 classes, ↓ from 0.608), genre F1=0.362, sub_genre F1=0.286, provenance F1=0.171
- provenance/sub_provenance produce identical results (1:1 parallel label columns, as predicted in Section 16.4)

### Track C — SAE: Plan Finalized (April 2026)
Implementation plan at `src/sae/plan/PLAN.md`. Key decisions:
- **Pre-trained SAE:** Arditi (2024) for Qwen2.5-7B-Instruct, 131k features, layers 7/15/23
- **Last-token pooling only** — SAE expects per-token activations, not mean-pooled
- **Pipeline:** Extract SAE features from existing activations (CPU) → sparse probing → feature analysis + probe direction decomposition
- **No steering** for now — deferred as more complex with unclear value for classification
- **No mean pooling** — SAE wasn't trained on averaged vectors
- **Dataset path refactor** deferred until 40k dataset arrives
- Can start implementation on letters corpus immediately (no GPU needed for SAE extraction)

## Updated SEAL Data (2026-04-14) — Round 5 Re-run Complete

Chunrong sent corrected CSVs on 2026-04-14 with partially-resolved period labels. **All phases 0→A→B→C have been re-run on the new data (2026-04-14).**

| Change | Old | New |
|--------|-----|-----|
| seal.csv: `Middle Babylonian/Assyrian` (65 frags) | compound | split: `Middle Babylonian` (24) + `Middle Assyrian` (6) + ambiguous remaining (35) |
| dll.csv: `Neo-Assyrian and Late Babylonian` (44 frags) | compound (entire corpus) | split: `Neo-Assyrian` (18) + `Neo or Late Babylonian` (26 still ambiguous) |
| lbpl.csv | unchanged | unchanged |

**Phase C re-run results (2026-04-14):**

| Task | N | Classes | F1m (tier0) | F1m (maximal) | Change vs round 4 |
|------|--:|--------:|------------:|--------------:|-------------------|
| `period` | 383 | 9 | 0.473 | 0.352 | ↓ from 0.608/0.464 (more classes) |
| `genre` | 384 | 16 | 0.362 | 0.269 | unchanged |
| `sub_genre` | 246 | 43 | 0.286 | 0.267 | unchanged |
| `provenance` | 374 | 25 | 0.171 | 0.122 | negligible change |
| `sub_provenance` | 374 | 25 | 0.171 | 0.122 | negligible change |
| `domain` | 384 | 3 | 0.952 | 0.876 | negligible change |

All 12 runs: p=0.001. Signal is real for all tasks. Period F1m drop is expected — 9 classes vs 6, with several new small classes (Middle Assyrian=6, Archaic=2) scoring F1=0.

## Research Direction Clarification (Nathan Wasserman, 2026-04-13)

Nathan clarified the real research goal: cross-period separation (OB vs LB vs NA) is too
easy — he can do it himself. The hard problem is **fine-grained chronological ordering
within a period** (e.g., ordering 229 OB texts among themselves within the ~500-year OB
window). Period labels are 500-year buckets based on expert judgment, not hard science.
This reframes the thesis contribution — cross-period results are a sanity check; the deeper
question needs finer sub-period labels from Chunrong (still pending).

## Blocked On
- Chunrong finishing period re-labeling (some tablets still being checked)
- Fine-grained within-period sub-labels (new requirement from Nathan)
- Full dataset delivery from advisor (40k+ Akkadian fragments)

## Next Steps
1. ~~**Re-run SEAL Phases 0→C** on updated CSVs from Chunrong~~ ✅ Done 2026-04-14 (round 5 re-run complete).
2. **SEAL Phase D — Linear probing**: modify `01_extract_activations.py` + `02_linear_probe.py`, run on cluster.
3. **SEAL Phase E — Documentation**: update README, run log, commit each phase.
4. **Track C — SAE implementation** (can run in parallel):
   - Verify SAE loading on cluster (sae-lens + Arditi weights)
   - Extract SAE features at layers 7/15/23 (last_token, tier0+maximal)
5. Receive full dataset from advisor; re-run linear probe + SAE on full dataset.
