#!/bin/bash
#SBATCH --job-name=WB1d_llama70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/logs/WB1d_llama70b_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# WB1d — CELL-B entity extraction, Llama-2-70B and its materialized random twin
# (the twin must already exist from W0_build_random_llama70b).
#   0 llama2_70b   1 llama2_70b_random
#   sbatch v_1/src/world_models/akkadian/sbatch/WB1d_extract_llama70b.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== WB1d extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/akkadian/extract_entity.py --method "$M" --batch-size 32 \
    || { echo "FAILED extract ${M}"; exit 1; }

commit_push "WB1d: cell-B entity acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/assyrian_ruler/metadata.json \
    v_1/src/world_models/activations/${M}/mesopotamian_place/metadata.json
echo "=== Done ${M}. $(date) ==="
