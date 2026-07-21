#!/bin/bash
#SBATCH --job-name=W1c_extract_llama
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/W1c_extract_llama_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# W1c — Llama-2 7B/13B, trained + random-init (from_config, seed 42), gpu:1 each.
#   0 llama2_7b   1 llama2_13b   2 llama2_7b_random   3 llama2_13b_random
# meta-llama repos are gated: set HF_TOKEN, or rely on the automatic
# NousResearch/Llama-2-*-hf fallback (same weights, ungated).
#   sbatch v_1/src/world_models/sbatch/W1c_extract_llama.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_7b llama2_13b llama2_7b_random llama2_13b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== W1c extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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

commit_push "W1c: world-models acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
