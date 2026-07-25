#!/bin/bash
#SBATCH --job-name=WB1_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/WB1_extract_%A_%a.out
#SBATCH --array=0-6
# ===========================================================================
# WB1 — CELL-B entity-level extraction, gpu:1 ladder. Both datasets inner-loop.
# Only 354 short rows per model, so this is minutes per arm, not hours.
#   0 qwen3_1b7  1 qwen3_8b  2 qwen3_32b(stride 2)
#   3 thalesian_akk300m  4 thalesian_cunei400m  5 umt5_base
#   6 random (random-init Qwen3-8B control)
# Four pooling sites are written: ent_last, ent_mean, last, mean.
# gpt-oss = WB1b, llamas = WB1c/WB1d.  Needs WB0 to have committed the CSVs.
#   sbatch v_1/src/world_models/akkadian/sbatch/WB1_extract_entity.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== WB1 extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/akkadian/extract_entity.py --method "$M" \
    || { echo "FAILED extract ${M}"; exit 1; }

commit_push "WB1: cell-B entity acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/assyrian_ruler/metadata.json \
    v_1/src/world_models/activations/${M}/mesopotamian_place/metadata.json
echo "=== Done ${M}. $(date) ==="
