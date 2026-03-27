# Linear Probing Pipeline — Execution Plan

## Current State (2026-03-26)

### Completed
- **Step 00 (Tokenization Check)** — ✅ Job 2030. Qwen2.5-7B handles Akkadian fine. 0 unknown tokens, mean 267 tokens/text.
- **Step 00b (Quick EDA) — first run** — ✅ Job 2031. Mean-pooled final-layer embeddings extracted, PCA + t-SNE plot saved. BUT this was the old version (mean pooling only).
- **Step 00b (Quick EDA) — re-run** — ⏳ Job 2052 submitted with updated code (both mean + last-token pooling). Currently pending/running on cluster.

### Not Yet Done
- Review 00b plots locally
- Step 01 (Extract all-layer activations) — 4 runs: 2 cleanings × 2 poolings, ~8 hours
- Step 02 (Linear probe at every layer)
- Step 03 (Analyze results, classify outcome A/B/C)

### Key Decisions Made
- **Model:** `Qwen/Qwen2.5-7B-Instruct` (28 layers, hidden dim 3584). Llama-2-7b gated access not yet resolved.
- **Pooling:** Both mean-pooling AND last-token pooling (added by user request)
- **Large .npz files stay on cluster only** (gitignored). Plots and JSON results go in git.

### Known Fixes Applied
- `utils.py`: `full_text` → `text` column rename, period label mapping (`'Neo-Assyrian'` → `'NA'` etc.)
- All sbatch files: switched from `meta-llama/Llama-3.1-8B-Instruct` to `Qwen/Qwen2.5-7B-Instruct`
- `00b_quick_eda.py`: now produces 2 plots (mean + last_token pooling)
- `01_extract_activations.py`: added `--pooling {mean,last_token}` arg
- `01_extract.sh`: runs 4 extractions (2 cleanings × 2 poolings), 8 hour time limit

---

## Step-by-Step Commands

### Workflow Pattern (repeats for each step)
```
LOCAL: edit code → git push
CLUSTER: git pull → sbatch → squeue → cat logs
CLUSTER: git add results → git commit → git push (plots/JSON only, not .npz)
LOCAL: git pull → review results
```

### Check Job Status (cluster)
```bash
squeue -u $USER          # PD=pending, R=running, CG=cancelling, gone=done
scancel <job_id>         # cancel a job
```

### Read Job Logs (cluster)
```bash
ls v_1/src/linear_probing/logs/                           # find log filename
cat v_1/src/linear_probing/logs/<name>_<job_id>.out       # read output
```

Log file naming pattern:
- `tok_check_<id>.out` — step 00
- `quick_eda_<id>.out` — step 00b
- `extract_<id>.out` — step 01
- `probe_<id>.out` — step 02
- `analyze_<id>.out` — step 03

---

## Remaining Steps In Order

### 1. Wait for Step 00b to finish, review plots

**On cluster:**
```bash
squeue -u $USER
# when job disappears:
cat v_1/src/linear_probing/logs/quick_eda_2052.out
```

**Push results from cluster to git:**
```bash
cd ~/projects/HUJI-THESIS--YARIN
git add v_1/src/linear_probing/results/plots/ v_1/src/linear_probing/results/tokenization_check.json
git commit -m "Add step 00/00b results from cluster"
git push
```

**On local Mac:**
```bash
cd /Users/yarin.b/git/lititure-review
git pull
# review plots at:
# v_1/src/linear_probing/results/plots/quick_eda_final_layer_mean.png
# v_1/src/linear_probing/results/plots/quick_eda_final_layer_last_token.png
```

**What to look for in the plots:**
- Do OB/NA/LB form separate clusters? → good sign, proceed to step 01
- No clustering at all? → still proceed (mid-layers may differ), but temper expectations
- One period overlaps heavily with another? → note which pair, interesting for later analysis

### 2. Run Step 01 — Extract All-Layer Activations (~8 hours)

**On cluster:**
```bash
cd ~/projects/HUJI-THESIS--YARIN
git pull
sbatch v_1/src/linear_probing/sbatch/01_extract.sh
```

This runs 4 extractions sequentially:
1. tier0 + mean pooling
2. maximal + mean pooling
3. tier0 + last_token pooling
4. maximal + last_token pooling

**Verify when done:**
```bash
cat v_1/src/linear_probing/logs/extract_<job_id>.out
# check for "Done!" at the end, no errors
# verify files exist:
ls v_1/src/linear_probing/results/activations/qwen2.5-7b-instruct/tier0/
ls v_1/src/linear_probing/results/activations/qwen2.5-7b-instruct/maximal/
ls v_1/src/linear_probing/results/activations/qwen2.5-7b-instruct/tier0_last_token/
ls v_1/src/linear_probing/results/activations/qwen2.5-7b-instruct/maximal_last_token/
```

Expected: 29 `.npz` files per directory (layers 00–28) + `metadata.json`

### 3. Run Step 02 — Linear Probe (~2 hours, CPU only)

**IMPORTANT:** Before running, the `02_linear_probe.py` script needs to be updated to handle both pooling methods. Currently it only reads from `tier0/` and `maximal/` directories. It needs to also probe `tier0_last_token/` and `maximal_last_token/`.

**On cluster:**
```bash
sbatch v_1/src/linear_probing/sbatch/02_probe.sh
```

**Verify when done:**
```bash
cat v_1/src/linear_probing/logs/probe_<job_id>.out
ls v_1/src/linear_probing/results/plots/
# expected: layer_accuracy_curve.png, confound_random_label.png, tsne_by_layer.png, confusion_matrix_best_layer.png
```

**Push results:**
```bash
git add v_1/src/linear_probing/results/plots/ v_1/src/linear_probing/results/probe_results_*.json
git commit -m "Add step 02 probe results"
git push
```

### 4. Run Step 03 — Analyze & Classify Outcome (~30 min, CPU only)

**On cluster:**
```bash
sbatch v_1/src/linear_probing/sbatch/03_analyze.sh
```

**Verify when done:**
```bash
cat v_1/src/linear_probing/logs/analyze_<job_id>.out
cat v_1/src/linear_probing/results/summary_qwen2.5-7b-instruct.json
```

**Push results:**
```bash
git add v_1/src/linear_probing/results/
git commit -m "Add step 03 analysis results"
git push
```

### 5. Review Locally

**On Mac:**
```bash
git pull
```

Review:
- `results/summary_qwen2.5-7b-instruct.json` — outcome A/B/C
- `results/plots/` — all plots
- Update `results/PIPELINE_RUN_LOG.md` with findings

---

## File Locations

### On Local Mac
```
/Users/yarin.b/git/lititure-review/v_1/src/linear_probing/
├── utils.py, 00_*.py, 01_*.py, 02_*.py, 03_*.py  (scripts)
├── sbatch/                                          (sbatch files)
├── plan/                                            (PLAN.md, this file, etc.)
└── results/
    ├── PIPELINE_RUN_LOG.md                          (running summary)
    ├── tokenization_check.json                      (step 00 output)
    ├── plots/                                       (all plots, synced via git)
    └── .gitignore                                   (excludes .npz files)
```

### On Cluster
```
~/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/
├── logs/                                            (slurm output files)
└── results/
    ├── activations/qwen2.5-7b-instruct/
    │   ├── tier0/           (29 .npz + metadata.json)
    │   ├── maximal/         (29 .npz + metadata.json)
    │   ├── tier0_last_token/    (29 .npz + metadata.json)
    │   └── maximal_last_token/  (29 .npz + metadata.json)
    └── plots/                   (synced via git)
```

---

## Cluster Reference

| Command | What |
|---------|------|
| `squeue -u $USER` | Check job status |
| `scancel <id>` | Cancel a job |
| `sbatch <file>.sh` | Submit a job |
| `cat v_1/src/linear_probing/logs/<name>_<id>.out` | Read job output |
| `ls v_1/src/linear_probing/logs/` | Find log filenames |

## Future: Switching to Llama-2-7b

When Llama access is resolved:
1. Accept license at `huggingface.co/meta-llama/Llama-2-7b-hf`
2. Get new token, store on cluster: `echo "hf_xxx" > ~/.cache/huggingface/token`
3. Change model in all sbatch files: `Qwen/Qwen2.5-7B-Instruct` → `meta-llama/Llama-2-7b-hf`
4. Re-run entire pipeline (results save to separate `llama-2-7b-hf/` directories)
