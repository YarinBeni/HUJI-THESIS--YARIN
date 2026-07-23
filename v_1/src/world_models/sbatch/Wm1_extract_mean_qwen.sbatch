#!/bin/bash
#SBATCH --job-name=Wm1_mean_qwen
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/Wm1_mean_qwen_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# Wm1 — add the `mean` pooling to the English decoders (they were extracted `last`
# only; encoders already have mean). Re-extract with --sites last,mean; the existing
# last.*.npz are rewritten identically and mean.*.npz added. gpu:1 each.
#   0 qwen3_1b7  1 qwen3_8b  2 qwen3_32b  3 random   (random = random-init Qwen3-8B)
#   sbatch v_1/src/world_models/sbatch/Wm1_extract_mean_qwen.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Wm1 extract ${M} (last,mean) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u v_1/src/world_models/extract_acts.py --method "$M" --sites last,mean \
    || { echo "FAILED extract ${M}"; exit 1; }
commit_push "Wm1: world-models acts ${M} +mean (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
