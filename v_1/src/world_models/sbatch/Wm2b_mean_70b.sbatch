#!/bin/bash
#SBATCH --job-name=Wm2b_mean_70b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=v_1/src/world_models/logs/Wm2b_mean_70b_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# Wm2b — LEAN re-run of the hung Llama-70B mean probes: --sites mean skips re-doing
# the already-committed `last`, and the bigger mem avoids the thrash that stalled
# 16644_6/_9 at 11h+. Kill those first:  scancel 16644_6 16644_9
#   0 llama2_70b   1 llama2_70b_random
#   sbatch v_1/src/world_models/sbatch/Wm2b_mean_70b.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Wm2b mean-only ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python -u v_1/src/world_models/probe_wm.py --method "$M" --sites mean \
    || { echo "FAILED probe ${M}"; exit 1; }
commit_push "Wm2b: ${M} mean probe (lean re-run, job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/projections/${M}
echo "=== Done ${M}. $(date) ==="
