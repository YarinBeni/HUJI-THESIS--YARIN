#!/bin/bash
#SBATCH --job-name=C5b_heads
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=chrono/sbatch/logs/C5b_heads_%A_%a.out
# ===========================================================================
# C5b — re-run the erase_provenance head (seed 0, 5 folds) per encoder with
# head saving on, so the nonlinear-recovery check can load the trained heads.
# Heads (state_dicts) are copied into tracked chrono/reports/tier0/heads/.
# ===========================================================================
set -uo pipefail
ENCS=(cunei400m llama2_7b qwen3_8b); CFG="emin2_${ENCS[$SLURM_ARRAY_TASK_ID]}_t0_akk_erase_provenance"
echo "=== C5b ${CFG} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/heads
ART=chrono/artifacts_tier0
for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py --config "chrono/configs/${CFG}.yaml" --seed 0 --fold "${FOLD}" \
        --corpus "${ART}/corpus_chrono.parquet" --views "${ART}/views.parquet" \
        --ruler-table "${ART}/ruler_table.parquet" --store-root "${ART}/emb_store" --splits-dir "${ART}/splits" \
        || { echo "FAILED train_cjb ${CFG} fold ${FOLD}"; exit 1; }
done
cp chrono/artifacts/heads/"${CFG}"-s0-*.pt chrono/reports/tier0/heads/ 2>/dev/null || true
ls -la chrono/reports/tier0/heads | tail -6
commit_push_sandbox "C5b: saved heads ${CFG} (job ${SLURM_JOB_ID})" chrono/reports/tier0/heads
echo "=== Done. $(date) ==="
