#!/bin/bash
#SBATCH --job-name=W1d_extract_llama70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=v_1/src/world_models/logs/W1d_extract_llama70b_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# W1d — Llama-2-70B trained (0) + random-init (1). bf16 ~137GB -> gpu:4 with
# device_map=auto. Task 1 loads the checkpoint W0 materialized — RUN W0 FIRST
# (or submit with:  sbatch --dependency=afterok:<W0-jobid> ...).
# Layer stride 2 (41 of 80 layers incl. the last).
#   sbatch v_1/src/world_models/sbatch/W1d_extract_llama70b.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== W1d extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WM_MODELS_DIR="${WM_MODELS_DIR:-$HOME/projects/wm_models}"

python -u v_1/src/world_models/extract_acts.py --method "$M" --batch-size 32 \
    || { echo "FAILED extract ${M}"; exit 1; }

commit_push "W1d: world-models acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
