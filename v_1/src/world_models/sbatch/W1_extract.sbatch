#!/bin/bash
#SBATCH --job-name=W1_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/W1_extract_%A_%a.out
#SBATCH --array=0-6
# ===========================================================================
# W1 — G&T-dataset activation extraction, one gpu:1 model per array task, all six
# entity datasets inner-loop. gpt-oss = W1b (gpu:8), llamas = W1c/W1d.
#   0 qwen3_1b7   1 qwen3_8b   2 qwen3_32b (stride 2)
#   3 thalesian_akk300m   4 thalesian_cunei400m   5 umt5_base
#   6 random (Qwen3-8B from_config, seed 42 — thesis-ladder control)
# npz are gitignored/cluster-local; committed = metadata.json per dataset.
#   sbatch v_1/src/world_models/sbatch/W1_extract.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== W1 extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/extract_acts.py --method "$M" \
    || { echo "FAILED extract ${M}"; exit 1; }

commit_push "W1: world-models acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
