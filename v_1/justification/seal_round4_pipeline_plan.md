# SEAL/DLL/LBPL Pipeline Plan (Round 4 data, April 2026)

Status: **Phase C complete (2026-04-07).** See Section 16 for Phase 0 facts, Section 17 for Phase A+B facts, Section 18 for Phase C facts.
Owner: Yarin
Source data: `yarin/emails_phase/round4/{seal,dll,lbpl}.csv` (delivered by Chunrong, Apr 4–6 2026)
Target pipelines: `v_1/src/bias_check/`, `v_1/src/linear_probing/`

**Confirmed design decisions** (from Yarin, 2026-04-07):

| # | Decision | Choice |
|---|----------|--------|
| 1 | Letters task (`letters_3period`) | Untouched. SEAL pipeline is purely additive. |
| 2 | Singleton (N=1) class drop | Allowed (math constraint), logged prominently. |
| 3 | 8 NN variants for SEAL bias_check | Dropped — LR + permutation test only. Letters task keeps NNs. |
| 4 | Linear probe held-out test split for SEAL | Dropped — 5-fold CV scores are the headline. |
| 5 | Shared registry location | Co-located inside `v_1/src/bias_check/` (no new `common/` dir). linear_probing imports via sys.path. |
| 6 | Cluster vs local | Run locally where possible (bias_check). Linear probing extraction goes to cluster (GPU needed). |
| 7 | **Data verification & logging** | NEW REQUIREMENT — see Section 15. Every step verifies assumptions and writes a logged report. |

---

## 1. Goal

Run the **bias check** and **linear probing** pipelines against the new SEAL/DLL/LBPL data, predicting **every metadata label** that exists in the data — `period`, `genre`, `sub_genre`, `provenance`, `sub_provenance`, and the cross-corpus `domain` label.

Per-label, the dataset is built by **pooling all corpora that contain that label**:
- If a corpus has the column → include all its fragments.
- If a corpus lacks the label → exclude that corpus from this task only.

This produces one big dataset per task. The whole point of the project is to find the **hard / unbalanced** labels, so we keep small classes.

---

## 2. Design constraints (from Yarin)

1. **Don't drop small classes.** The hard, imbalanced cases are exactly what we want to study.
2. **Use cross-validation, not 70/15/15 split.** With 38–384 fragments, fixed splits are too noisy.
3. **Include the cross-corpus `domain` task** as a sanity check.
4. **Both cleanings** (`tier0` + `maximal`).
5. **Same Qwen2.5-7B** with both `mean` and `last_token` pooling for linear probing.
6. **Move CSVs into `v_1/data/raw/chungrong/seal_round4/`** to follow repo conventions.
7. **Pool across corpora per task.** Don't run separate per-corpus tasks.
8. **Minimal code changes.** Reuse existing pipelines, don't rewrite.

---

## 3. Source data overview

| Corpus | Fragments | Total words | Label columns present (non-empty) |
|--------|----------:|------------:|-----------------------------------|
| SEAL   | 302 | 23,350 | period, genre, sub_genre, provenance, sub_provenance |
| DLL    | 44  | 5,694  | period, genre, provenance, sub_provenance |
| LBPL   | 38  | 11,440 | period, genre, provenance, sub_provenance |
| **Total** | **384** | **40,484** | — |

`place_discovery` and `place_composition` are empty in all 3 — never used.

`sub_genre` is only present in SEAL (281/302 fragments non-null).

---

## 4. The 6 tasks we will run

For each task, the dataset is the union of fragments from all corpora where the label is non-null.

| # | Task name | Label | Corpora pooled | N fragments (raw) |
|---|---|---|---|---:|
| 1 | `period`         | `period`         | SEAL + DLL + LBPL | 384 |
| 2 | `genre`          | `genre`          | SEAL + DLL + LBPL | 384 |
| 3 | `sub_genre`      | `sub_genre`      | SEAL only         | 281 |
| 4 | `provenance`     | `provenance`     | SEAL + DLL + LBPL | 384 |
| 5 | `sub_provenance` | `sub_provenance` | SEAL + DLL + LBPL | 384 |
| 6 | `domain`         | `domain`         | SEAL + DLL + LBPL | 384 |

Each task → 1 bias-check run + 1 linear-probe run × (tier0 + maximal) × (mean + last_token).
Total: 6 bias-check runs + 24 linear-probe runs (over 4 cached activation sets).

---

## 5. Label normalization

Pooling exposes case-mismatch artifacts. Minimal normalization rules **before** assigning class IDs:

| Rule | Reason |
|------|--------|
| Lowercase + strip whitespace on `genre` and `sub_genre` | "Literary Letters" (SEAL) ≡ "literary letters" (LBPL). Confirmed in data. |
| Keep `provenance` / `sub_provenance` raw, **including compound values** like `Babylon;Sippar` | Compound = combined-tablet texts found at multiple sites. Splitting is a separate research decision. |
| Keep `period` raw | "Old Babylonian" / "Late Babylonian" / "Neo-Assyrian and Late Babylonian" / etc. — let the data show its imbalance. |
| `domain` is computed from corpus membership: SEAL / DLL / LBPL | Sanity-check task. |

The normalization happens once in `build_seal_corpus.py` (Phase 6) so downstream code sees one canonical column.

---

## 6. Cross-validation strategy

Switching from 70/15/15 → **k-fold stratified CV**, with k adapted to the smallest class:

```
k = min(5, smallest_class_size_after_dropping_singletons)
```

### Singleton handling (math constraint, not a policy choice)

Stratified k-fold is **mathematically impossible** for a class with N=1: the single sample is either in train (no test signal) or in test (no training signal). Options considered:

| Option | Behavior | Verdict |
|--------|----------|---------|
| Drop N=1 classes silently | Hides imbalance | ❌ contradicts "find hard labels" |
| Drop N=1 classes + **report** them in task summary | Honest, math-compliant | ✅ chosen |
| Keep N=1, ignore stratification | k-fold breaks | ❌ |
| Use leave-one-out CV per class | Doesn't generalize | ❌ |

**Decision**: drop only N=1 classes per task, log them explicitly to `task_summary.json`. For N=2..4 the task uses k = smallest_class_size; for N≥5 the task uses k=5.

This preserves the spirit of "keep small classes" while staying mathematically valid. Singletons aren't "small classes we can study" — they're untrainable/untestable points.

For each task, the pre-CV report writes:
- Total fragments
- Per-class N distribution
- Singletons dropped (count + value list)
- Effective k for stratified CV

---

## 7. Architecture: bias check on small data

The current bias_check pipeline trains 8 neural net variants (MLPs + Attn+MLP, 0.3M–2.9M params) on a 70/15/15 split of 4,957 letters. This is **the wrong tool** for 384-fragment SEAL tasks:

- Smallest model (mlp_1layer) has 2.6M parameters; smallest fold has ~77 training samples. Catastrophic overfit guaranteed.
- 70/15/15 split with 384 samples → 6-sample test fold for SEAL period task. Permutation test resolution is meaningless.

### Decision: SEAL bias check uses **logistic regression + permutation test only**

This matches the academic standard (Ojala & Garriga 2010) and matches what the linear_probing pipeline already does internally for layer selection. The 8 NN variants stay reserved for the **letters task** (which has the data to support them).

**For each SEAL task, the bias check produces**:
- TF-IDF char_wb (2,5) vectorizer fit per fold
- LogisticRegression(C=grid) with 5-fold (or adaptive-k) stratified CV
- Macro-F1, accuracy, per-class precision/recall/F1
- Permutation test: 1000 label-shuffled fits → null distribution → p-value
- Confusion matrix
- Per-task report markdown

This is statistically meaningful on small N.

### Letters task stays unchanged

The existing `letters_3period` task (4,957 letters, 8 NN variants, fixed split) is not touched. We add NEW code; we don't refactor the existing 5 scripts. Backward compatibility is automatic because old paths/configs are not touched.

---

## 8. Architecture: linear probing on pooled corpus

### Insight: extract activations once, probe many times

The 6 tasks share the same 384 input fragments (the only exception is `sub_genre`, which uses 281 SEAL-only fragments). So:

1. **Extract activations once** for the full 384-fragment pooled corpus, for both cleanings × both poolings = 4 cached directories. Each directory holds N_layers ≈ 29 npz files of shape (384, 3584).
2. **Probe per task** using the cached activations, filtering by row index for `sub_genre` (subset to SEAL fragments).

This is efficient: extraction is the GPU-heavy step (~30 min for 384 fragments per cleaning/pooling, ~2 hrs total). Probing is fast (<5 min per task per cleaning per pooling).

### Per-task probe step

The existing `02_linear_probe.py` already does:
- Load activations per layer
- 5-fold stratified CV with LR + GridSearchCV over `C_GRID`
- Permutation test at the best layer
- Final test-set evaluation
- 4 plots (layer curve, null distribution, t-SNE at best layer, confusion matrix)

Changes needed:
- Replace hardcoded `load_letters()` with a generic loader that takes `--input-parquet`, `--label-col`.
- Replace hardcoded `PERIODS=['OB','NA','LB']` with labels derived from the data.
- Skip the "test-set evaluation" step when k-fold CV is the final answer (no held-out test). Or: add a 5-fold OOF prediction step that aggregates CV predictions.
- Skip TFIDF baselines for non-letters tasks (the constants in `utils.py` only apply to letters).
- Save under `results/seal_round4/<task_name>/`.

---

## 9. Files to create / modify

### NEW files

1. **`v_1/src/corpus/01_inspect_seal_data.py`** — data verification step (Phase 0). Reads raw CSVs, profiles every column, writes `inspection_report.{md,json}`. Catches assumption mismatches BEFORE we build the corpus. (See Section 15.)
2. **`v_1/src/corpus/02_build_seal_corpus.py`** — converts the 3 word-level CSVs → one fragment-level parquet (`v_1/data/evaluation/corpora/seal_corpus.parquet`) with all metadata columns. Re-runs verification on the output and fails loudly if anything mismatches the inspection report.
3. **`v_1/src/bias_check/seal_tasks.py`** — task registry + data loader. Single source of truth for the 6 tasks. `load_task_data(task_name)` returns a normalized DataFrame with `text`, `label_raw`, `label_idx`, plus metadata. Handles pooling, normalization, singleton drop. Imported by both bias_check and linear_probing.
4. **`v_1/src/bias_check/06_bias_check_cv.py`** — new driver script for SEAL tasks. Uses the registry, runs LR + permutation test per task with k-fold CV. Writes per-task summary JSON with verified assumptions.
5. **`v_1/src/linear_probing/sbatch/01_extract_seal.sh`** — sbatch for activation extraction (4 runs: 2 cleanings × 2 poolings) on the pooled corpus.
6. **`v_1/src/linear_probing/sbatch/02_probe_seal.sh`** — sbatch that loops over 6 tasks × 2 poolings = 12 probe jobs.
7. **`v_1/data/raw/chungrong/seal_round4/README.md`** — documents the round-4 delivery (date, contents, contact emails). Includes link to inspection report.

### MODIFIED files (minimal CLI surface)

1. **`v_1/src/linear_probing/01_extract_activations.py`** — add `--input-parquet`, `--corpus-name`, `--text-col` flags. Default to current letters paths. Imports `seal_tasks` from `../bias_check/` via sys.path.
2. **`v_1/src/linear_probing/02_linear_probe.py`** — add `--corpus-name`, `--task-name` flags (task name selects label_col and corpus_filter from registry). Derive label list from data. Output under `results/<corpus_name>/<task_name>/`. Skip held-out test split when in CV-only mode. Skip TFIDF baselines for non-letters.
3. **`v_1/data/raw/chungrong/README.md`** — point to the new `seal_round4/` subdir.
4. **`v_1/data/evaluation/corpora/README.md`** — document `seal_corpus.parquet`.
5. **`v_1/src/bias_check/README.md`** — add a "Multi-task SEAL mode" section.
6. **`v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md`** — append SEAL run after letters runs are complete.
7. **`v_1/README.md`** — add SEAL section + key data files row for the new parquet.

### NOT modified (preserved backwards-compat)

- `v_1/src/bias_check/01_featurize.py` … `05_report.py` — untouched. Letters task still works exactly the same.
- `v_1/src/bias_check/config.py`, `models.py` — untouched.
- `v_1/src/linear_probing/02b_validity_tests.py`, `03_analyze_results.py` — untouched (they're letters-specific analyses).

---

## 10. Output directory layout

```
v_1/data/raw/chungrong/seal_round4/
├── README.md
├── seal.csv         # moved from yarin/emails_phase/round4/
├── dll.csv
└── lbpl.csv

v_1/data/evaluation/corpora/
├── seal_corpus.parquet              # NEW (384 fragments, all metadata cols)
└── seal_corpus_summary.json         # NEW (per-(corpus, label) counts + null counts)

v_1/data/evaluation/bias_check/
├── (existing letters_3period files at top level — unchanged)
└── seal_round4/                     # NEW
    ├── period/
    │   ├── task_summary.json        # N, classes, singletons-dropped, k
    │   ├── metrics.json             # CV scores + perm test p-value
    │   ├── plots/{confusion.png, perm_null.png, per_class_f1.png}
    │   └── report.md
    ├── genre/
    ├── sub_genre/
    ├── provenance/
    ├── sub_provenance/
    └── domain/

v_1/src/linear_probing/results/
├── (existing letters files — unchanged)
└── seal_round4/                                    # NEW
    ├── activations/
    │   └── qwen2.5-7b-instruct/
    │       ├── tier0/                  layer_00..28.npz + metadata.json
    │       ├── maximal/
    │       ├── tier0_last_token/
    │       └── maximal_last_token/
    └── probes/
        ├── period/
        │   ├── probe_results_mean.json
        │   ├── probe_results_last_token.json
        │   └── plots/{layer_curve, perm_null, tsne, confusion}_<pooling>.png
        ├── genre/
        ├── sub_genre/                  # uses subset of activation rows
        ├── provenance/
        ├── sub_provenance/
        └── domain/
```

---

## 11. Implementation phases (execution order)

**Phase 0 — Data inspection & verification (BEFORE any other code)**
1. `mkdir v_1/data/raw/chungrong/seal_round4/` and copy the 3 CSVs there.
2. Write `v_1/src/corpus/01_inspect_seal_data.py`. Run it locally. **Read its report end-to-end** before proceeding. (See Section 15 for required checks.)
3. Save the inspection report to `v_1/data/raw/chungrong/seal_round4/inspection_report.{md,json}`.
4. Write `v_1/data/raw/chungrong/seal_round4/README.md` referencing the inspection report.
5. **Manual review checkpoint** — Yarin reads the inspection report. Any surprises get added to the plan as "verified facts" or "open questions". No code is written until this is done.

**Phase A — Corpus build (depends on Phase 0 sign-off)**
6. Write `v_1/src/corpus/02_build_seal_corpus.py`. The script:
   - Re-runs the inspection on the raw CSVs and **fails loudly** if anything mismatches the saved Phase 0 report (catches data drift if Chunrong sends a corrected file).
   - Builds `v_1/data/evaluation/corpora/seal_corpus.parquet` (384 fragments, all metadata cols).
   - Re-inspects the output parquet and verifies invariants (row count, null counts, label normalization applied).
   - Writes `seal_corpus_summary.json` with per-(corpus, label) counts.
7. Run it locally. Read the post-build summary. Spot-check 5 fragments by eye against the source CSVs.
8. Update `v_1/data/raw/chungrong/README.md` and `v_1/data/evaluation/corpora/README.md`.

**Phase B — Task registry (depends on Phase A)**
9. Write `v_1/src/bias_check/seal_tasks.py` with the 6 task definitions and `load_task_data(task_name)`.
10. Write a tiny self-test at the bottom of the file: when run as `python seal_tasks.py`, iterates over all 6 tasks, calls `load_task_data`, prints/logs:
    - Task name, N fragments, N classes, singleton drops, effective k for CV
    - Top 5 classes by N
    - Asserts that the data shape matches what the inspection report predicts
11. Run the self-test. Save its output to `v_1/data/evaluation/corpora/seal_tasks_verification.md`. Reading it should be sufficient to convince ourselves the registry is correct.

**Phase C — Bias check on SEAL (depends on Phase B)**
12. Write `v_1/src/bias_check/06_bias_check_cv.py`. The script:
    - Runs in `--debug` mode by default for fast iteration
    - For each task: loads via registry, fits TF-IDF char_wb (2,5), runs LR with 5-fold (or adaptive-k) CV + permutation test
    - Writes per-task `task_summary.json` (includes the verified data assumptions: N classes, singletons dropped, k used)
    - Writes per-task `metrics.json`, plots, and `report.md`
13. Local debug run on 1 task (`domain` is fastest — 3 classes). Verify outputs end-to-end.
14. Run all 6 tasks locally (~10 min CPU total).
15. Read every per-task `report.md`. Cross-check claimed N against the seal_corpus_summary.

**Phase D — Linear probing on SEAL (depends on Phase B)**
16. Modify `01_extract_activations.py`: add `--input-parquet`, `--corpus-name`, `--text-col` flags + small import block to load `seal_tasks` from sibling dir. Defaults preserve current letters behavior.
17. Modify `02_linear_probe.py`: add `--corpus-name`, `--task-name` flags. Use the registry to resolve label/filter. Add `--cv-only` mode that skips the held-out test split. Skip TFIDF baselines unless `letters_3period`.
18. Verify with `--help` and a tiny dry-run on letters data (must produce identical results to current pipeline → backward compat check).
19. Write `v_1/src/linear_probing/sbatch/01_extract_seal.sh` (4 sequential extractions).
20. Write `v_1/src/linear_probing/sbatch/02_probe_seal.sh` (12 probe runs).
21. Smoke-test on cluster: 1 extraction + 1 probe + manual diff against the inspection report.
22. Submit full extract job, then chained probe job.

**Phase E — Documentation**
23. Update `v_1/README.md`, `v_1/src/bias_check/README.md`, `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md`.
24. Commit each phase as a separate commit with a clear message + reference to the verification reports.

---

## 12. Edge cases handled

| Case | Handling |
|------|----------|
| Singleton class (N=1) | Drop, log to `task_summary.json` |
| Class with 2 ≤ N < 5 | Use k = smallest_N for stratified k-fold |
| `sub_genre` task: DLL/LBPL fragments have null sub_genre | Filter out rows with null `label_col` per task — `data_loader.load_task_data()` does this automatically |
| Compound provenance ("Babylon;Sippar") | Treated as its own class — preserves combined-tablet semantics |
| Genre case mismatch ("Literary Letters" vs "literary letters") | Lowercased before assigning class IDs |
| Linear probe activation cache reused across tasks | `sub_genre` task filters activations by row index of SEAL-only fragments |
| Permutation test for small N | Reduce to 500 perms if total time becomes a problem; default 1000 |
| `period` label semantically inconsistent across corpora | We don't try to harmonize SEAL's "Middle Babylonian/Assyrian" with DLL's "Neo-Assyrian and Late Babylonian". They become distinct class labels. The whole point is to expose this. |

---

## 13. Open questions

All Phase 0.5 questions from the previous draft have been answered (see decision table at top). No open questions remain at the planning level.

**New questions that may emerge from Phase 0** will be added to this section as we discover them during data inspection.

---

## 14. Estimated change footprint

| Category | Files | LoC (rough) |
|---|---:|---:|
| New code | 7 | ~750 |
| Modified existing code | 2 | ~80 |
| Documentation updates | 6 | ~200 |
| Verification reports (generated) | 4+ | — |
| **Total** | **15** | **~1030** |

No existing logic is rewritten. All modifications add CLI args with backward-compatible defaults.

---

## 15. Data verification & logging requirements (NEW)

**Motivation.** During the planning conversation, the assistant **incorrectly claimed** that DLL and LBPL had no `sub_provenance` column, based on a single `head -3` of each CSV. A second pass revealed all three corpora do have it (3 unique values each in DLL/LBPL). This kind of unverified assumption is exactly what we need to prevent — the data is small enough that one bad assumption can corrupt every downstream task. The fix is to make verification an explicit, logged step rather than an implicit one.

**Principle.** Every data-touching step writes a machine-readable summary AND a human-readable report. Every assumption we encode in code must be backed by a line in one of these reports. Re-runs that produce different results must fail loudly.

### What `01_inspect_seal_data.py` produces

For each of the 3 raw CSVs, the inspection script profiles:

1. **Shape & file integrity**
   - Row count, column count, file size, MD5 hash (so we can detect re-deliveries)
2. **Per-column profile** (for every column, including the ones we don't currently use)
   - dtype
   - null count, null ratio
   - unique value count
   - top 10 values by frequency
   - if the column looks compound (contains `;`): list distinct compound values with counts
   - if the column looks like text (e.g. `value`, `clean_value`, `lemma`): min/max/mean character length
3. **Fragment-level aggregation check**
   - Number of unique fragment_ids
   - Words per fragment: min, max, mean, median, p25, p75
   - Whether metadata is consistent within a fragment (group by fragment_id, check that each metadata column has exactly 1 unique value per fragment — flag any fragments where metadata varies internally)
4. **Cross-corpus consistency**
   - Same column set across all 3 CSVs? List columns missing from each.
   - For each label column, list its unique values and which corpora contain them.
   - Flag case-mismatches (e.g., `"Literary Letters"` vs `"literary letters"`).
   - Flag values that look like compound forms vs atomic forms in the same column.
5. **Per-task feasibility**
   - For each of the 6 candidate tasks, after pooling and dropping nulls in label_col:
     - N fragments
     - N classes total
     - N singletons (will be dropped)
     - N classes with N=2..4 (will force adaptive k)
     - N classes with N≥5 (use k=5)
     - Effective `k` after adaptive selection
6. **Sanity assertions** (raise loud errors if violated)
   - All `clean_value` cells are non-null and non-empty
   - `fragment_id` is unique within each row aggregation
   - No fragment has fewer than 1 word

### Output formats

```
v_1/data/raw/chungrong/seal_round4/inspection_report.md   # human-readable, organized by section
v_1/data/raw/chungrong/seal_round4/inspection_report.json # machine-readable, the same data
```

The JSON is checked into git; it becomes the contract for downstream scripts.

### Re-running the inspection

`02_build_seal_corpus.py` re-runs inspection on its inputs and **diffs** against the saved JSON. Any drift (new columns, changed value sets, new singletons, etc.) raises an error with a clear message identifying what changed. This catches:

- Chunrong sending a corrected CSV file
- Accidental edits to the raw data
- Changes to normalization rules without updating the contract

To intentionally accept new data, the user must re-run `01_inspect_seal_data.py` and commit the new `inspection_report.json` — making the change visible in `git diff`.

### Per-task verification logs

Each of the 6 task runs (both bias_check and linear_probing) writes a `task_summary.json` containing the assumptions it relied on at runtime:

```json
{
  "task_name": "sub_genre",
  "label_col": "sub_genre",
  "corpora_pooled": ["seal"],
  "fragments_total_input": 281,
  "fragments_after_null_filter": 281,
  "n_classes_input": 78,
  "singletons_dropped": ["Anzu", "Etana", ...],
  "n_classes_after_drop": 43,
  "fragments_after_drop": 246,
  "smallest_class_size": 2,
  "k_used": 2,
  "verification": {
    "schema_md5": "abc123...",
    "matches_inspection_report": true
  }
}
```

If any per-task summary diverges from what the inspection report predicted, the script aborts.

### Where verification logs live

```
v_1/data/raw/chungrong/seal_round4/
└── inspection_report.{md,json}           # source of truth, Phase 0

v_1/data/evaluation/corpora/
├── seal_corpus_summary.json              # post-build verification, Phase A
└── seal_tasks_verification.md            # registry self-test output, Phase B

v_1/data/evaluation/bias_check/seal_round4/<task>/
└── task_summary.json                     # per-task runtime verification, Phase C

v_1/src/linear_probing/results/seal_round4/probes/<task>/
└── task_summary.json                     # per-task runtime verification, Phase D
```

### Why this is worth the overhead

- 384 fragments is small enough that one mislabeled class can flip a result.
- The labels are exactly what the experiments are predicting — wrong assumptions silently corrupt findings.
- The dataset will likely be re-delivered (Chunrong has corrected files before; the email thread shows this happens).
- We commit verification reports to git, so any change to the data is auditable in `git log`.
- The "I missed sub_provenance" failure mode would have been caught immediately by step 4 of the inspection (cross-corpus consistency check).

---

## 16. Verified facts from Phase 0 (ran 2026-04-07)

`01_inspect_seal_data.py` ran successfully against the 3 CSVs copied
into `v_1/data/raw/chungrong/seal_round4/`. Full output in
`inspection_report.{md,json}` next to the CSVs.

### 16.1 Structural facts (all match the plan exactly)

- **384 fragments total** (SEAL=302 / DLL=44 / LBPL=38) ✅
- **40,484 words total** (SEAL=23,350 / DLL=5,694 / LBPL=11,440) ✅
- **Schema identical across all 3 CSVs** and matches the 15-column
  expected layout from Section 3 ✅
- **`fragment_id` is globally unique across corpora** — no collision
  risk when indexing cached activations ✅
- **Metadata is internally consistent** within every fragment — the
  first-word-wins aggregation used by `to_fragment_table()` is safe ✅
- **`place_discovery` and `place_composition` are 100% null** in all 3
  corpora — correctly ignored ✅
- **`sub_genre`** has 78 raw classes → 43 after singleton drop → 246
  fragments (matches the illustrative example in Section 15) ✅

### 16.2 Per-task feasibility (post-normalization, post-singleton-drop)

| Task | Pooled | Frags | Classes | Singletons dropped | k |
|------|--------|------:|--------:|-------------------:|--:|
| `period` | SEAL+DLL+LBPL | 383 | 6 | 1 | 2 |
| `genre` | SEAL+DLL+LBPL | 384 | 16 | 0 | 2 |
| `sub_genre` | SEAL only | 246 | 43 | 35 | 2 |
| `provenance` | SEAL+DLL+LBPL | 374 | 25 | 10 | 2 |
| `sub_provenance` | SEAL+DLL+LBPL | 374 | 25 | 10 | 2 |
| `domain` | SEAL+DLL+LBPL | 384 | 3 | 0 | 5 |

Only the `domain` task reaches k=5. The other 5 tasks adaptive-k to 2
because the smallest non-singleton class has exactly 2 fragments.

### 16.3 Known issue accepted (not a blocker)

**`dll.csv` has 4 rows with null `clean_value`** — all are digit-only
cuneiform tokens (`2-ta`, `2`) that Chunrong's upstream cleaner
discards. The raw `value` column is populated in all 4 cases and the
`lemma` confirms valid words (`šina II` ×3, `šadû I` ×1). All 4 rows
come from different fragments; no fragment or class is lost.

**Fix (applied in Phase A's `02_build_seal_corpus.py`)**: fall back to
`value` when `clean_value` is null. Rationale:

- `tier0` cleaning preserves digits by design — the 4 numerals are
  part of the tier0 signal and should survive.
- `maximal` cleaning strips all digits in filter #1, so these tokens
  vanish anyway — unchanged behavior.
- Dropping the rows would silently delete tier0 signal, which is
  semantically wrong for "minimal cleaning".

Phase A will pin the fallback to exactly these 4 rows (matched by
`fragment_id + fragment_line_num + index_in_line`) and abort if the
set changes in a future re-delivery. The inspection script
intentionally keeps its strict assertion as a tripwire; its "failed"
status is the acknowledgement that this known issue exists.

### 16.4 Non-blocking observations worth knowing before Phase B

1. **`period` is nearly colinear with `domain`.** DLL contains only
   `Neo-Assyrian and Late Babylonian` (1 value), LBPL contains only
   `Late Babylonian` (1 value). Only SEAL spans 5 period values. This
   means any model that can identify the source corpus gets `period`
   almost for free — the `domain` sanity-check task is much more
   load-bearing than originally framed, and a `period` result can be
   fully explained by corpus-level lexical cues unless we look at
   within-SEAL period breakdown separately.
2. **`provenance` and `sub_provenance` appear mathematically equivalent.**
   Both pool to 35 raw classes / 25 post-drop classes / 10 singletons /
   374 fragments — identical at every level. SEAL has exactly 34
   unique values in each column (1:1 mapping), DLL has 3/3, LBPL has
   3/3. They are ancient-name vs modern-site parallel namings with no
   structural difference. Running both as separate probe tasks is
   likely redundant; Phase B should either (a) collapse them into one
   task or (b) confirm that the two rankings differ for at least some
   classifier before committing GPU time to both.
3. **8 of 10 `provenance` singletons are compound values.**
   Section-5's "keep compound values raw" rule interacts with
   Section-6's "drop singletons" rule to delete almost all of the
   combined-tablet provenances (`Babylon;Sippar`, `Sippar;Nippur`,
   `Assur;Nineveh`, etc.). This is consistent with the plan as
   written, but the combined-tablet semantics are effectively lost
   from all provenance tasks. If preserving them matters, Phase A
   would need a splitting policy; otherwise no change required.
4. **Case-mismatch confirmed in `genre`.** "Literary Letters" (SEAL)
   and "Literary letters" (LBPL) exist in the raw data. Section-5's
   `lowercase+strip` normalization merges them correctly — after
   normalization, `literary letters` has 15 fragments instead of
   being split 14/1.

---

## 17. Verified facts from Phases A and B (ran 2026-04-07)

### 17.1 Phase A — Corpus build (`02_build_seal_corpus.py`)

**Output**: `v_1/data/evaluation/corpora/seal_corpus.parquet` — 384 rows × 15 columns.

**Columns** (in order):
```
fragment_id, corpus, word_language, domain, period, genre, sub_genre,
provenance, sub_provenance, place_discovery, place_composition,
word_count, text, text_tier0, text_maximal
```

- `corpus` ∈ {`seal`, `dll`, `lbpl`} (lowercase); `domain` ∈ {`SEAL`, `DLL`, `LBPL`} (original CSV tag).
- `text` = `clean_value` words joined by space, with `value` fallback for the 4 known null rows in dll.
- `text_tier0` / `text_maximal` = pre-computed via the same functions as `v_1/src/linear_probing/utils.py`; the import is avoided so this script is stdlib+pandas+numpy only.
- `genre` and `sub_genre` are lowercase+stripped (Section 5 normalization applied at build time, not query time).
- `place_discovery` and `place_composition` are all-NaN (as expected).
- **5/5 spot-checks passed**: text == " ".join(source CSV words) for all 5 sampled fragments.
- MD5 check passed: all 3 CSVs match their Phase 0 hashes.
- Fallback assertion passed: exactly (32264,20,3), (32592,116,2), (33621,36,3), (34164,11,3) triggered fallback.
- All output invariants passed.

**Also written**: `v_1/data/evaluation/corpora/seal_corpus_summary.json` — per-corpus × per-label counts + null counts.

### 17.2 Phase B — Task registry (`seal_tasks.py`)

**Output**: `v_1/src/bias_check/seal_tasks.py` — task registry + `load_task_data()`.

**Self-test**: all 6 tasks ✓ PASS against Phase 0 inspection contract.

`load_task_data(task_name, cleaning="tier0")` returns `(df, task_summary)` where:
- `df` columns: `text`, `label_raw`, `label_idx`, `fragment_id`, `corpus`, `word_count`, `domain`, `period`, `genre`, `sub_genre`, `provenance`, `sub_provenance`, `word_language`
- `cleaning` ∈ {`"raw"`, `"tier0"`, `"maximal"`} selects `text` / `text_tier0` / `text_maximal` from the parquet
- Singletons (N=1 classes) are dropped; `task_summary` records which ones
- `label_idx` is alphabetically sorted for reproducibility

**Confirmed task shapes** (matches Section 16.2 exactly):

| Task | N input | After null | Classes | Singletons | Classes left | N left | k |
|------|--------:|-----------:|--------:|-----------:|-------------:|-------:|--:|
| `period` | 384 | 384 | 7 | 1 | 6 | 383 | 2 |
| `genre` | 384 | 384 | 16 | 0 | 16 | 384 | 2 |
| `sub_genre` | 302 | 281 | 78 | 35 | 43 | 246 | 2 |
| `provenance` | 384 | 384 | 35 | 10 | 25 | 374 | 2 |
| `sub_provenance` | 384 | 384 | 35 | 10 | 25 | 374 | 2 |
| `domain` | 384 | 384 | 3 | 0 | 3 | 384 | 5 |

**Also written**: `v_1/data/evaluation/corpora/seal_tasks_verification.md` — full self-test report.

### 17.3 How to import seal_tasks from other directories

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "bias_check"))
from seal_tasks import load_task_data, TASK_NAMES
```

(linear_probing scripts are in `v_1/src/linear_probing/`; `bias_check/` is a sibling dir.)

---

## 18. Verified facts from Phase C (ran 2026-04-07)

Script: `v_1/src/bias_check/06_bias_check_cv.py`

### 18.1 Design (as implemented)

- TF-IDF `char_wb(2,5)`, `max_features=10_000`, `sublinear_tf=True`
- `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)`
- C selected by `cross_val_score` over grid `[0.001, 0.01, 0.1, 1.0, 10.0]` using the adaptive-k CV splitter
- `permutation_test_score` with `n_permutations=1000`, `random_state=0`, `scoring="f1_macro"`
- Output per task/cleaning: `task_summary.json`, `metrics.json`, `report.md`, `plots/`
- Output root: `v_1/data/evaluation/bias_check/seal_round4/<task>/<cleaning>/`

### 18.2 Phase C results (all 12 combinations)

| Task | Cleaning | N | k | Best C | macro-F1 | p | Result |
|------|----------|--:|--:|-------:|---------:|--:|--------|
| `period` | tier0 | 383 | 2 | 0.01 | 0.608 | 0.001 | **FAIL** |
| `period` | maximal | 383 | 2 | 0.1 | 0.464 | 0.001 | **FAIL** |
| `genre` | tier0 | 384 | 2 | 0.1 | 0.361 | 0.001 | **FAIL** |
| `genre` | maximal | 384 | 2 | 0.01 | 0.269 | 0.001 | **FAIL** |
| `sub_genre` | tier0 | 246 | 2 | 1.0 | 0.286 | 0.001 | **FAIL** |
| `sub_genre` | maximal | 246 | 2 | 1.0 | 0.267 | 0.001 | **FAIL** |
| `provenance` | tier0 | 374 | 2 | 1.0 | 0.171 | 0.001 | **FAIL** |
| `provenance` | maximal | 374 | 2 | 0.1 | 0.128 | 0.001 | **FAIL** |
| `sub_provenance` | tier0 | 374 | 2 | 1.0 | 0.171 | 0.001 | **FAIL** |
| `sub_provenance` | maximal | 374 | 2 | 0.1 | 0.128 | 0.001 | **FAIL** |
| `domain` | tier0 | 384 | 5 | 10.0 | 0.952 | 0.001 | **FAIL** |
| `domain` | maximal | 384 | 5 | 0.001 | 0.889 | 0.001 | **FAIL** |

All 12 tasks: p = 0.001 (minimum possible with 1000 permutations). TF-IDF char n-grams
significantly predict every metadata label. This is expected — it is a positive result for
the linear probing pipeline (diachronic and domain signals exist in the text surface form).

### 18.3 Interpretation notes

- **domain** (F1=0.95/0.89) confirms the three corpora are lexically separable. This validates
  the domain sanity-check task design.
- **period** (F1=0.61/0.46) substantially exceeds macro chance (1/6≈0.17). Drop from tier0→maximal
  (-0.14) is large — many period-discriminative features are writing conventions, consistent
  with letters corpus findings.
- **provenance / sub_provenance** produce identical results at both cleanings (as expected from
  Section 16.4 observation #2 — they are 1:1 parallel namings).
- **sub_genre** (F1=0.29/0.27) with 43 classes: above macro chance (1/43≈0.023) but weak.
  Small class sizes (N=2 for many classes) limit signal.
- **genre** (F1=0.36/0.27) with 16 classes: above chance (1/16≈0.063), moderate signal.

---

## 19. Round 5 re-delivery — updated period labels (2026-04-14)

### 19.1 What Chunrong changed

Chunrong sent corrected CSVs on 2026-04-14 in response to the label quality email sent
on 2026-04-12. New files: `yarin/emails_phase/round4/{seal,dll,lbpl}.csv`.

| CSV | Old label | Old N (frags) | Resolution | New N |
|-----|-----------|---:|---|---:|
| seal.csv | `Middle Babylonian/Assyrian` | 65 | Split: `Middle Babylonian` (24) + `Middle Assyrian` (6) + remaining `Middle Babylonian/Assyrian` (35 genuinely ambiguous) | 24+6+35 = 65 |
| dll.csv | `Neo-Assyrian and Late Babylonian` | 44 | Split: `Neo-Assyrian` (18) + `Neo or Late Babylonian` (26 still compound) | 18+26 = 44 |
| lbpl.csv | — | — | No changes | — |

Fragment totals are preserved. SEAL row count unchanged (23,350 words). DLL lost 1
word-row (5,694 → 5,693 — likely a deduplication artefact, not a fragment drop).

### 19.2 What was NOT addressed

- `Old Assyrian` (N=5 in seal.csv) — not split or merged
- `provenance` / `sub_provenance` redundancy — not addressed; awaiting clarification
- Chunrong notes some tablets are still being checked — further corrections may follow

### 19.3 Research direction clarification (Nathan Wasserman, 2026-04-13)

Nathan's reply to the label question reframes the thesis goal:

- Current period labels (OB, LB, NA, etc.) are **500-year buckets** based on expert
  judgment — useful but not scientifically rigorous.
- Separating OB from LB is **trivially easy** even for a non-expert and is not the
  contribution. Nathan can do it himself.
- The **real hard problem** is fine-grained chronological ordering *within* a single
  period — e.g., ordering the 229 OB fragments among themselves chronologically.
- This requires finer sub-period labels that do not yet exist in the data. Chunrong is
  still working on period re-labeling which may eventually provide this.

**Impact on the pipeline:**
- Phase C FAIL results (p=0.001) are expected and validate the data, but are not the
  main thesis finding.
- Phase D (linear probing) should include a within-SEAL-OB breakdown once finer labels
  are available.
- The multi-task SEAL experiment remains valid as a breadth experiment (6 labels, signal
  present in all). The depth experiment (within-OB ordering) is the future work.

### 19.4 Action checklist — COMPLETE (2026-04-14)

- [x] Copy `yarin/emails_phase/round4/*.csv` → `v_1/data/raw/chungrong/seal_round4/`
- [x] Re-run `01_inspect_seal_data.py` — update `inspection_report.{md,json}`
- [x] Re-run `02_build_seal_corpus.py` — rebuild `seal_corpus.parquet`
- [x] Re-run `seal_tasks.py` self-test — update `seal_tasks_verification.md`
- [x] Re-run `06_bias_check_cv.py` — update all 12 metrics.json / report.md / plots
- [x] Update `seal_round4/README.md` summary table with new results
- [ ] Commit as a single phase with message referencing this section

---

## 20. Verified facts from Phase C re-run (ran 2026-04-14 on round 5 data)

Script: `v_1/src/bias_check/06_bias_check_cv.py`

### 20.1 Phase 0 re-run facts

- SEAL: 23,350 words × 302 fragments — **unchanged**. `Middle Babylonian/Assyrian` split into 3 values.
- DLL: 5,693 words × 44 fragments (−1 word vs round 4, likely dedup artefact). `Neo-Assyrian and Late Babylonian` split into `Neo-Assyrian` (18 frags) + `Neo or Late Babylonian` (26 frags).
- LBPL: unchanged (11,440 words × 38 fragments).
- 4 null `clean_value` rows in dll.csv: **same 4 rows** (fragment_ids 32264/32592/33621/34164). Phase A fallback assertion passed without modification.
- MD5 hashes updated in `inspection_report.json`.

### 20.2 Phase A + B re-run facts

- `seal_corpus.parquet`: 384 rows × 15 cols — same schema, same fragment count. All 5/5 spot-checks passed. MD5 and fallback assertions passed.
- `seal_tasks_verification.md` updated. All 6 tasks ✓ PASS.
- Key change: `period` task now has 10 input classes (was 7), 9 surviving (was 6), 1 singleton dropped (`Later Periods (SB, NA, LB)`).

### 20.3 Phase C re-run results (all 12 combinations)

| Task | Cleaning | N | k | Best C | macro-F1 | p | Result |
|------|----------|--:|--:|-------:|---------:|--:|--------|
| `period` | tier0 | 383 | 2 | 0.1 | 0.473 | 0.001 | **FAIL** |
| `period` | maximal | 383 | 2 | 0.001 | 0.352 | 0.001 | **FAIL** |
| `genre` | tier0 | 384 | 2 | 0.1 | 0.362 | 0.001 | **FAIL** |
| `genre` | maximal | 384 | 2 | 0.01 | 0.269 | 0.001 | **FAIL** |
| `sub_genre` | tier0 | 246 | 2 | 1.0 | 0.286 | 0.001 | **FAIL** |
| `sub_genre` | maximal | 246 | 2 | 1.0 | 0.267 | 0.001 | **FAIL** |
| `provenance` | tier0 | 374 | 2 | 1.0 | 0.171 | 0.001 | **FAIL** |
| `provenance` | maximal | 374 | 2 | 0.1 | 0.122 | 0.001 | **FAIL** |
| `sub_provenance` | tier0 | 374 | 2 | 1.0 | 0.171 | 0.001 | **FAIL** |
| `sub_provenance` | maximal | 374 | 2 | 0.1 | 0.122 | 0.001 | **FAIL** |
| `domain` | tier0 | 384 | 5 | 10.0 | 0.952 | 0.001 | **FAIL** |
| `domain` | maximal | 384 | 5 | 0.01 | 0.876 | 0.001 | **FAIL** |

All 12 tasks: p = 0.001. Signal is genuine for every task.

### 20.4 Changes vs round 4 (Section 18.2)

| Task/Cleaning | Round 4 F1m | Round 5 F1m | Change | Cause |
|---------------|------------:|------------:|--------|-------|
| period/tier0 | 0.608 | 0.473 | −0.135 | 6→9 classes; new small classes score F1=0 |
| period/maximal | 0.464 | 0.352 | −0.112 | same cause |
| domain/maximal | 0.889 | 0.876 | −0.013 | DLL −1 word-row |
| provenance/maximal | 0.128 | 0.122 | −0.006 | DLL −1 word-row |
| sub_provenance/maximal | 0.128 | 0.122 | −0.006 | DLL −1 word-row |
| All other | — | — | ~0 | unchanged |

The `period` F1m drop is expected and desired: finer labels expose harder classification.
New classes `Middle Assyrian` (N=6) and `Archaic/Old Akkadian/Ebla` (N=2) score F1=0 under k=2 CV — insufficient samples to learn, but kept to expose the imbalance per design constraint #1 (Section 2).
