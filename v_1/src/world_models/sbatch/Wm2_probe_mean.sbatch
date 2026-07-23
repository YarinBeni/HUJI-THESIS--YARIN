#!/bin/bash
#SBATCH --job-name=Wm2_probe_mean
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/logs/Wm2_probe_mean_%A_%a.out
#SBATCH --array=0-9
# ===========================================================================
# Wm2 — re-probe the English decoders after the mean acts landed (Wm1/Wm1c/Wm1d).
# probe_wm auto-detects sites on disk, so this writes the new {dataset}.mean.ridge.json
# alongside the existing .last. CLEANUP OFF (keep both poolings). Run after extraction.
#   0 qwen3_1b7 1 qwen3_8b 2 qwen3_32b 3 random 4 llama2_7b 5 llama2_13b
#   6 llama2_70b 7 llama2_7b_random 8 llama2_13b_random 9 llama2_70b_random
#   sbatch v_1/src/world_models/sbatch/Wm2_probe_mean.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Wm2 probe ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/probe_wm.py --method "$M" \
    || { echo "FAILED probe ${M}"; exit 1; }
commit_push "Wm2: world-models probes ${M} +mean (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/projections/${M}
echo "=== Done ${M}. $(date) ==="
