# Parallel Execution Plans — SEAL Round 5 Re-run + EDA GUI
# Status: FINAL (2026-04-14)

Dependency order:

```
A ─────────────────────────────────────────────────────┐
                                                        ├──→ E (runs last)
B (no deps — start immediately) ──────────────────────►│
                                                        │
wait for A, then: C (rsync parquet → 4 sbatch jobs) ──►│
                                                        │
D-training (start immediately, ~10h) ─────────────────►│
D-extraction (wait for A + D-training, then run) ──────┘
```

Plans A, B, D-training can all start right now in parallel.
Plan C needs Plan A's updated parquet first.
Plan D-extraction needs both Plan A's parquet AND D-training to finish.
Plan E runs last, once you have at least B + C done (D optional at merge time).

---

## Plan A — Re-run pipeline on updated Chungrong CSVs (Local)

**CONTEXT:**
Akkadian thesis project. v_1/ is the working dir. Chungrong sent corrected CSVs with
fixed period labels on 2026-04-14. They are at yarin/emails_phase/round4/{seal,dll,lbpl}.csv
(repo-relative) and have NOT yet been copied into the pipeline. We need to re-run
Phases 0→C on the new data.

**Run all commands from the repo root** (`/Users/yarin.b/git/lititure-review`).

READ FIRST (in order):
1. `v_1/justification/seal_round4_pipeline_plan.md` — Sections 11, 17, 18, 19 (phase steps + verified facts + new data)
2. `v_1/PROGRESS.md` — current state
3. `v_1/data/raw/chungrong/seal_round4/README.md` — known issues + what changed in re-delivery

**STEPS:**

1. Copy new CSVs:
   ```
   cp yarin/emails_phase/round4/seal.csv v_1/data/raw/chungrong/seal_round4/seal.csv
   cp yarin/emails_phase/round4/dll.csv  v_1/data/raw/chungrong/seal_round4/dll.csv
   cp yarin/emails_phase/round4/lbpl.csv v_1/data/raw/chungrong/seal_round4/lbpl.csv
   ```

2. Re-run Phase 0 (updates data contract):
   ```
   python3 v_1/src/corpus/01_inspect_seal_data.py
   ```
   → overwrites `inspection_report.{md,json}` next to the CSVs.
   Check the diff of `inspection_report.json` against Section 19.1 of the pipeline plan:
   - SEAL: `Middle Babylonian/Assyrian` should split into 3 values; total SEAL frags unchanged.
   - DLL: `Neo-Assyrian and Late Babylonian` splits; DLL total drops from 5,694 → 5,693 words.
   - LBPL: no changes.
   If `dll.csv` now has different null-`clean_value` rows than the 4 known ones
   (fragment_ids 32264/32592/33621/34164), Phase A will abort — re-pin the fallback
   assertion in `02_build_seal_corpus.py` before continuing.

3. Re-run Phase A (rebuilds corpus parquet):
   ```
   python3 v_1/src/corpus/02_build_seal_corpus.py
   ```
   → overwrites `seal_corpus.parquet` + `seal_corpus_summary.json` in
   `v_1/data/evaluation/corpora/`.
   Note: this script re-reads `inspection_report.json` from disk (the freshly written
   version from step 2) and uses it as the new contract, so the diff-check passes.

4. Re-run Phase B self-test (updates task verification):
   ```
   python3 v_1/src/bias_check/seal_tasks.py
   ```
   → overwrites `v_1/data/evaluation/corpora/seal_tasks_verification.md`.
   N values should shift for `period` task (more classes after split).

5. Re-run Phase C (full bias check, all 12 combinations):
   ```
   python3 v_1/src/bias_check/06_bias_check_cv.py --plots
   ```
   → overwrites all `metrics.json`, `report.md`, `plots/` under
   `v_1/data/evaluation/bias_check/seal_round4/<task>/<cleaning>/`.

6. Read every `report.md`. Cross-check N values against `seal_tasks_verification.md`.
   Print new summary table (task × cleaning × F1m × p).

7. Update `v_1/data/evaluation/bias_check/seal_round4/README.md`:
   Replace the summary table with the new numbers.

8. Update `v_1/PROGRESS.md` and `v_1/justification/seal_round4_pipeline_plan.md`:
   - In Section 19.4 checklist, mark all items done.
   - Add Section 20 with new Phase C verified facts (same format as Section 18).

**OUTPUT FILES:**
- `v_1/data/raw/chungrong/seal_round4/{seal,dll,lbpl}.csv` (updated)
- `v_1/data/raw/chungrong/seal_round4/inspection_report.{md,json}` (updated)
- `v_1/data/evaluation/corpora/seal_corpus.parquet` (updated)
- `v_1/data/evaluation/corpora/seal_tasks_verification.md` (updated)
- `v_1/data/evaluation/bias_check/seal_round4/<task>/<cleaning>/{metrics,task_summary}.json` (updated ×12)
- `v_1/data/evaluation/bias_check/seal_round4/README.md` (updated summary table)

**DONE WHEN:** All 12 bias check runs complete, N values match verification doc, README updated.

---

## Plan B — Build HTML GUI shell (Local)

**CONTEXT:**
Akkadian thesis project. We are building an interactive HTML EDA tool for 384 SEAL/DLL/LBPL
fragments. The GUI is a single self-contained HTML file using Plotly.js — no server needed,
opens in browser. It reads one JSON data file (`seal_viz_data.json`) and shows interactive
scatter plots with controls to switch embedding method, layer, cleaning, dim reduction,
and color-by label. **This plan has no dependencies — start immediately.**

READ FIRST:
1. `v_1/data/evaluation/corpora/seal_corpus.parquet` — understand the 15 metadata columns
2. `v_1/data/evaluation/corpora/seal_tasks_verification.md` — understand the 6 label tasks

**JSON SCHEMA CONTRACT** (`seal_viz_data.json` — this is the format you must produce AND consume):
```json
{
  "fragments": [
    {
      "fragment_id": 1520,
      "corpus": "seal",
      "domain": "SEAL",
      "period": "Old Babylonian",
      "genre": "incantations",
      "sub_genre": "scorpions",
      "provenance": "Nippur",
      "word_count": 199,
      "text_snippet": "first 15 words of text_tier0"
    }
  ],
  "embeddings": {
    // key format: "{method}__{cleaning}__{layer}__{reduction}"
    // method:    tfidf | qwen | random | mlm
    // cleaning:  tier0 | maximal
    // layer:     L00..L28 for qwen/random; L00/L04/L08/L12/L16 for mlm; "na" for tfidf
    // reduction: tsne | pca
    "tfidf__tier0__na__tsne":     [[x,y], [x,y], ...],  // 384 pairs
    "tfidf__maximal__na__pca":    [[x,y], ...],
    "qwen__tier0__L00__tsne":     [[x,y], ...],
    "mlm__tier0__L08__pca":       [[x,y], ...]
  }
}
```

**STEPS:**

1. Create directory `v_1/src/viz/` and write `v_1/src/viz/01_compute_tfidf_coords.py`:
   - Loads `v_1/data/evaluation/corpora/seal_corpus.parquet`
   - Fits TF-IDF `char_wb(2,5)` on `text_tier0` and `text_maximal` separately
   - Runs t-SNE (`perplexity=30`, `max_iter=1000`, `random_state=42`) and PCA (2 components) on each
   - Builds `seal_viz_data.json` with fragments metadata + 4 TF-IDF embedding keys:
     `tfidf__tier0__na__tsne`, `tfidf__tier0__na__pca`,
     `tfidf__maximal__na__tsne`, `tfidf__maximal__na__pca`
   - Saves to `v_1/src/viz/seal_viz_data.json`
   - Dependencies: pandas, numpy, sklearn. No torch.
   - Use `max_iter=` not `n_iter=` (sklearn ≥1.5 API).

2. Run it:
   ```
   python3 v_1/src/viz/01_compute_tfidf_coords.py
   ```
   Verify: 384 entries in `fragments`, 4 keys in `embeddings`, each with 384 `[x,y]` pairs.

3. Write `v_1/src/viz/seal_eda.html`:

   **Controls (top bar):**
   - Method dropdown: TF-IDF | Qwen | Random Qwen | Yarin MLM
   - Layer slider: 0–28 (disabled + grayed out when TF-IDF selected; clamped to 0–16
     when MLM selected, showing only L00/L04/L08/L12/L16 as valid stops)
   - Cleaning toggle: tier0 | maximal
   - Dim reduction toggle: t-SNE | PCA
   - Color by dropdown: domain | period | genre | sub_genre | provenance

   **Main area:** Plotly scatter plot
   - One dot per fragment
   - Color = selected label column (distinct color per class, legend)
   - Hover tooltip: `fragment_id`, `corpus`, `period`, `genre`, `sub_genre`,
     `provenance`, `word_count`, `text_snippet`
   - If embedding key not in JSON: show "data not yet available" message inline
     (graceful degradation — GUI works with only TF-IDF while cluster work is pending)

   **Implementation notes:**
   - Pure HTML + JS, no build step, no server
   - Load `seal_viz_data.json` via `fetch()` from same directory
   - Use Plotly.js CDN
   - Run with: `python3 -m http.server 8000` from `v_1/src/viz/`, then open
     `localhost:8000/seal_eda.html`

4. Test with TF-IDF data:
   ```
   cd v_1/src/viz && python3 -m http.server 8000
   ```
   Open `localhost:8000/seal_eda.html`. Verify all 4 TF-IDF combinations render.
   Verify hover shows correct metadata. Verify color-by works for all 5 label columns.

**IMPORTANT:** `seal_viz_data.json` is the live data file shared with Plan E. Plan E's
merge step adds Qwen/MLM keys to this same file. **Do not re-run step 2 after Plan E
has merged Qwen/MLM data** — it will overwrite those keys with TF-IDF-only content.

**OUTPUT FILES:**
- `v_1/src/viz/01_compute_tfidf_coords.py`
- `v_1/src/viz/seal_eda.html`
- `v_1/src/viz/seal_viz_data.json` (TF-IDF coords only initially)

**DONE WHEN:** GUI opens in browser, TF-IDF scatter plots render, all controls work,
hover shows correct metadata, graceful "not yet available" for Qwen/MLM slots.

---

## Plan C — Cluster: Qwen + Random Qwen embeddings + 2D coords (Remote)

**CONTEXT:**
Akkadian thesis project. We need Qwen2.5-7B-Instruct activations (pretrained + random weights)
for 384 SEAL fragments, reduced to 2D per layer (29 layers, L00–L28) via t-SNE + PCA.
Output is a JSON file matching the `seal_viz_data.json` schema.
We do NOT pull raw activations locally — only the 2D coordinate JSON (~3 MB).

**DEPENDS ON:** Plan A must be complete (need updated `seal_corpus.parquet` on cluster).
Rsync parquet to cluster before starting steps 2–4.

READ FIRST on cluster:
1. `v_1/src/linear_probing/01_extract_activations.py` — existing extraction script (letters, pretrained)
2. `v_1/src/linear_probing/01b_extract_random_baseline.py` — existing random-weights script (letters)
3. `v_1/src/linear_probing/utils.py` — `clean_tier0`, `clean_maximal`, `mean_pool`, `last_token_pool`
4. `v_1/src/linear_probing/sbatch/` — existing sbatch scripts for reference
5. `v_1/justification/seal_round4_pipeline_plan.md` Section 8 (Phase D architecture)
6. `v_1/src/bias_check/seal_tasks.py` — `load_task_data()` for how SEAL data is loaded

**STEPS:**

1. Rsync updated parquet to cluster (run locally after Plan A finishes):
   ```
   rsync -av v_1/data/evaluation/corpora/seal_corpus.parquet \
     <cluster>:~/projects/lititure-review/v_1/data/evaluation/corpora/
   ```

2. Write `v_1/src/linear_probing/03_extract_seal_activations.py`:
   **New script** — do NOT modify `01_extract_activations.py` (keep letters pipeline intact).
   Base the structure on `01_extract_activations.py` but replace `load_letters()` with
   direct parquet loading. Key differences from 01:
   - Accepts `--input-parquet PATH` (seal_corpus.parquet) and `--text-col COLUMN`
     (text_tier0 or text_maximal)
   - Loads the parquet, uses the specified text column (already cleaned), no extra cleaning needed
   - Reuses `mean_pool`, `last_token_pool` from `utils.py`
   - Output dir: `results/seal_round4/activations/qwen_{tier0|maximal}/`
   - Layer files named `layer_00.npz` ... `layer_28.npz` (shape: 384 × 3584 float32)
   - Saves `metadata.json` alongside (n_texts=384, model_id, text_col, fragment_ids)

3. Write `v_1/src/linear_probing/03b_extract_random_seal_activations.py`:
   **New script** — mirror the `01` / `01b` pattern.
   Same as `03_extract_seal_activations.py` but initializes model with random weights
   (same pattern as `01b`: `AutoConfig.from_pretrained` + `AutoModelForCausalLM.from_config`,
   `torch.manual_seed(SEED)`). Output dir: `results/seal_round4/activations/random_{tier0|maximal}/`.

4. Write 4 sbatch scripts in `v_1/src/linear_probing/sbatch/seal/`:
   - `extract_qwen_tier0.sh`     → runs `03_extract_seal_activations.py --text-col text_tier0`
   - `extract_qwen_maximal.sh`   → runs `03_extract_seal_activations.py --text-col text_maximal`
   - `extract_random_tier0.sh`   → runs `03b_extract_random_seal_activations.py --text-col text_tier0`
   - `extract_random_maximal.sh` → runs `03b_extract_random_seal_activations.py --text-col text_maximal`
   Each requests 1×H100, 2h walltime. Use existing sbatch scripts as templates for cluster config.

5. Submit all 4 jobs. Monitor with `squeue`.

6. After all 4 complete, write `v_1/src/linear_probing/04_compute_2d_coords.py`:
   For each of the 4 activation dirs × 29 layers:
   - Load `layer_XX.npz` (shape: 384 × 3584)
   - Run t-SNE (`perplexity=30`, `max_iter=1000`, `random_state=42`) — use `max_iter=` not `n_iter=`
   - Run PCA (2 components, `random_state=42`)
   - Store under keys matching the schema:
     `qwen__tier0__L00__tsne`, `qwen__tier0__L00__pca`, ..., `qwen__tier0__L28__pca`
     `qwen__maximal__L00__tsne`, ..., `random__tier0__L00__tsne`, etc.
   - Collect all 232 keys (2 methods × 2 cleanings × 29 layers × 2 reductions).
   - Save to `results/seal_round4/seal_qwen_coords.json`
   - Validate: every key has exactly 384 `[x,y]` pairs, no NaN/Inf.

7. Run:
   ```
   python3 v_1/src/linear_probing/04_compute_2d_coords.py
   ```
   (CPU job, ~1–2 hrs for all 232 combinations)

8. Rsync JSON back to local:
   ```
   rsync -av <cluster>:~/projects/lititure-review/v_1/src/linear_probing/results/seal_round4/seal_qwen_coords.json \
     v_1/src/viz/seal_qwen_coords.json
   ```

**OUTPUT FILES (on cluster):**
- `v_1/src/linear_probing/03_extract_seal_activations.py`
- `v_1/src/linear_probing/03b_extract_random_seal_activations.py`
- `v_1/src/linear_probing/04_compute_2d_coords.py`
- `v_1/src/linear_probing/sbatch/seal/*.sh`
- `results/seal_round4/activations/{qwen,random}_{tier0,maximal}/layer_XX.npz` (×4×29)
- `results/seal_round4/seal_qwen_coords.json`

**OUTPUT FILE (local, after rsync):**
- `v_1/src/viz/seal_qwen_coords.json`

**DONE WHEN:** `seal_qwen_coords.json` on local machine, 232 keys, each 384 `[x,y]` pairs.

---

## Plan D — Cluster: Retrain Yarin's Akkadian MLM (Remote)

**CONTEXT:**
A custom 16-layer Akkadian MLM ("Simplified Aeneas Twin", ~22M params, d_model=384)
was trained in Dec 2025 and its checkpoints were deleted. We need to retrain it and
extract per-layer embeddings for the 384 SEAL fragments. Code is fully preserved.
The model exposes hidden states only at `ANALYSIS_LAYERS = [0, 4, 8, 12, 16]` (5 layers).

**Dependencies:**
- D-training (steps 1–3) has NO dependencies — start immediately.
- D-extraction (steps 4–6) needs Plan A's `seal_corpus.parquet` AND D-training to finish.

READ FIRST:
1. `v_1/src/archive/baseline_mlm/model.py` — AeneasForMLM architecture (16 layers, d_model=384)
2. `v_1/src/archive/baseline_mlm/02_train.py` — training script, `ANALYSIS_LAYERS`, checkpoint logic
3. `v_1/src/archive/baseline_mlm/01_prepare_data.py` — data prep
4. `v_1/models/baseline/training_stats.json` — previous training run (val loss 3.020)
5. `v_1/justification/seal_round4_pipeline_plan.md` Section 8 (embedding context)

**STEPS:**

1. On cluster, verify training data is available: `v_1/data/training_ready/` (≈32k fragments expected).
   Check: `python3 -c "import pandas as pd; print(len(pd.read_parquet('v_1/data/training_ready/train_fragments.parquet')))"`.
   If not on cluster, rsync from local first:
   ```
   rsync -av v_1/data/training_ready/ <cluster>:~/projects/lititure-review/v_1/data/training_ready/
   ```

2. Write `v_1/src/linear_probing/sbatch/seal/train_mlm.sh`:
   - 1×H100, 12h walltime (original took ~8hrs on different hardware; add buffer)
   - Runs: `python3 v_1/src/archive/baseline_mlm/02_train.py`
   - Saves checkpoints to `v_1/models/baseline_retrained/` (NOT `baseline/` — keep the old stats)
   - Target: match or beat previous val loss of 3.020

3. Submit job. Monitor loss curve (check training_stats.json or slurm log).
   Once complete, verify best checkpoint is saved at `v_1/models/baseline_retrained/baseline_best.pt`
   (that's the filename `02_train.py` actually writes).
   **Do NOT delete the checkpoint this time.**

4. Rsync `seal_corpus.parquet` to cluster (if not already there from Plan C):
   ```
   rsync -av v_1/data/evaluation/corpora/seal_corpus.parquet \
     <cluster>:~/projects/lititure-review/v_1/data/evaluation/corpora/
   ```

5. Write `v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py`:
   - Loads best checkpoint from `v_1/models/baseline_retrained/baseline_best.pt`
   - Loads 384 SEAL fragments from `seal_corpus.parquet` (columns: `text_tier0`, `text_maximal`)
   - Extracts hidden states at `ANALYSIS_LAYERS = [0, 4, 8, 12, 16]` for each text column
   - Uses mean pooling across sequence dimension (no pad tokens — MLM uses its own tokenizer)
   - Runs t-SNE (`perplexity=30`, `max_iter=1000`, `random_state=42`) and PCA (2 components)
     on each (layer, cleaning) combination — 5 × 2 = 10 combinations
   - Saves to `results/seal_round4/seal_mlm_coords.json`
   - Key format: `mlm__tier0__L00__tsne`, `mlm__tier0__L04__pca`, `mlm__maximal__L16__tsne`, etc.
     (exactly 5 layers × 2 cleanings × 2 reductions = 20 keys)
   - Note: uses the Akkadian tokenizer from `v_1/data/training_ready/vocab.json` (set in `01_prepare_data.py`),
     NOT the Qwen tokenizer.

6. Run:
   ```
   python3 v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py
   ```

7. Rsync JSON back to local:
   ```
   rsync -av <cluster>:.../results/seal_round4/seal_mlm_coords.json \
     v_1/src/viz/seal_mlm_coords.json
   ```

**OUTPUT FILES:**
- `v_1/src/linear_probing/sbatch/seal/train_mlm.sh`
- `v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py`
- `v_1/models/baseline_retrained/` (checkpoints — do NOT delete)
- `v_1/src/viz/seal_mlm_coords.json` (after rsync, on local)

**Layer note for GUI (Plan E):** MLM has only 5 valid layer positions (L00, L04, L08, L12, L16).
The GUI's layer slider should show "not available" for positions L01–L03, L05–L07, L09–L11,
L13–L15, L17–L28 when MLM is selected. This is handled by Plan B's graceful degradation logic.

**DONE WHEN:** `seal_mlm_coords.json` on local machine, 20 keys
(`mlm × tier0+maximal × L00/L04/L08/L12/L16 × tsne+pca`), each 384 `[x,y]` pairs.

---

## Plan E — Merge all outputs into final GUI (Local, runs last)

**CONTEXT:**
All parallel tracks are complete. We have:
- `v_1/src/viz/seal_viz_data.json`    — metadata + TF-IDF coords (from Plan B)
- `v_1/src/viz/seal_qwen_coords.json` — Qwen + Random Qwen coords (from Plan C)
- `v_1/src/viz/seal_mlm_coords.json`  — Yarin MLM coords (from Plan D — optional)
We merge them into one final `seal_viz_data.json` and verify the GUI works end-to-end.

**IMPORTANT:** After this plan runs, do NOT re-run Plan B's step 2
(`01_compute_tfidf_coords.py`) — it will overwrite `seal_viz_data.json` and erase
Qwen/MLM keys.

READ FIRST:
1. `v_1/src/viz/seal_viz_data.json` — current state (TF-IDF only, 4 keys)
2. `v_1/src/viz/seal_qwen_coords.json` — Qwen/Random structure (232 keys)
3. `v_1/src/viz/seal_mlm_coords.json` — MLM structure (20 keys), if available
4. `v_1/src/viz/seal_eda.html` — GUI code, to understand what keys it expects

**STEPS:**

1. Write `v_1/src/viz/02_merge_coords.py`:
   - Load `seal_viz_data.json` (base — has `fragments` array + TF-IDF `embeddings` dict)
   - Load `seal_qwen_coords.json` — add all 232 keys to `embeddings` dict
   - Load `seal_mlm_coords.json` if file exists — add all 20 keys
   - Validate: every key has exactly 384 `[x,y]` pairs
   - Validate: no NaN or Inf values in any coordinate
   - Save merged result back to `seal_viz_data.json`
   - Print summary: total embedding keys, methods present, any missing expected keys

2. Run:
   ```
   python3 v_1/src/viz/02_merge_coords.py
   ```
   Verify summary output: expect ≥236 keys (4 tfidf + 232 qwen), 256 if MLM included.

3. Test GUI end-to-end:
   ```
   cd v_1/src/viz && python3 -m http.server 8000
   ```
   Open `localhost:8000/seal_eda.html`.
   Test every control combination:
   - All available methods render
   - Layer slider responds correctly per method (disabled for TF-IDF, clamped to
     L00/L04/L08/L12/L16 for MLM, L00–L28 for Qwen/Random)
   - Both cleanings render
   - Both dim reductions render
   - All 5 color-by options render with correct labels
   - Hover shows correct metadata
   - No browser console errors

4. Update `v_1/PROGRESS.md`: mark GUI complete, list final embedding keys available.

**OUTPUT FILES:**
- `v_1/src/viz/02_merge_coords.py`
- `v_1/src/viz/seal_viz_data.json` (final, all methods merged)

**DONE WHEN:** GUI renders all methods, all controls work, no console errors.
