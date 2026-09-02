#!/bin/bash
#SBATCH --job-name=C4_ladder
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=chrono/sbatch/logs/C4_ladder_%A_%a.out
# ===========================================================================
# C4 — P1 single-variable LEACE erasure ladder on frozen tier0 features (CPU):
# {cunei400m, llama2_7b, qwen3_8b} x {akk, eng}, ridge probe, 7 rungs each.
# Answers "if not the ruler's name, what does the representation date by?".
# ===========================================================================
set -uo pipefail
CONFIGS=(emin2_cunei400m_t0_akk emin2_cunei400m_t0_eng emin2_llama2_7b_t0_akk
         emin2_llama2_7b_t0_eng emin2_qwen3_8b_t0_akk emin2_qwen3_8b_t0_eng)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "=== C4 ladder ${CFG} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/ladder
ART=chrono/artifacts_tier0
python -u chrono/scripts/erasure_ladder.py --config "chrono/configs/${CFG}.yaml" --probe ridge \
    --corpus "${ART}/corpus_chrono.parquet" --splits-dir "${ART}/splits" \
    --store-root "${ART}/emb_store" --out-dir chrono/reports/tier0/ladder/scores \
    --table-out "chrono/reports/tier0/ladder/${CFG}_ridge.md" \
    || { echo "FAILED ladder ${CFG}"; exit 1; }
commit_push_sandbox "C4: erasure ladder ${CFG} (job ${SLURM_JOB_ID})" chrono/reports/tier0/ladder
echo "=== Done. $(date) ==="
