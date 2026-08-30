#!/bin/bash
#SBATCH --job-name=C3_emin
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=chrono/sbatch/logs/C3_emin_%A_%a.out
# ===========================================================================
# C3 — E-MIN: chrono-barlow training over the cached thalesian features
# (needs C1's EmbStore + C2's gate verdict). Array task = one seed (0..4);
# each task runs all 5 gkf_ruler folds sequentially — the adapter head is
# tiny, the folds are minutes each, and one task per (seed, fold) would
# waste 20 extra scheduler slots. Results rows land in the gitignored
# chrono/artifacts/results.parquet; the small results.parquet + per-run
# score parquets are copied into tracked chrono/reports/ before
# commit_push_sandbox (chrono/artifacts is gitignored by design).
# ===========================================================================
set -uo pipefail
SEED=${SLURM_ARRAY_TASK_ID}
echo "=== C3 emin seed ${SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox
mkdir -p chrono/sbatch/logs

for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py \
        --config chrono/configs/emin_thalesian.yaml \
        --seed "${SEED}" --fold "${FOLD}" \
        || { echo "FAILED train_cjb seed ${SEED} fold ${FOLD}"; exit 1; }
done

mkdir -p chrono/reports/scores
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
cp chrono/artifacts/scores/emin_thalesian-s"${SEED}"-*.parquet \
    chrono/reports/scores/ 2>/dev/null || true
commit_push_sandbox "C3: emin thalesian seed ${SEED} (job ${SLURM_JOB_ID})" \
    chrono/reports
echo "=== Done. $(date) ==="
