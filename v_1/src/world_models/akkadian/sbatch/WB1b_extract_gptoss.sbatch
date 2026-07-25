#!/bin/bash
#SBATCH --job-name=WB1b_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/WB1b_gptoss_%j.out
# ===========================================================================
# WB1b — CELL-B entity extraction for gpt-oss-120B (needs the 8-GPU shard).
#   sbatch v_1/src/world_models/akkadian/sbatch/WB1b_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== WB1b extract ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
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

commit_push "WB1b: cell-B entity acts ${M} (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/activations/${M}/assyrian_ruler/metadata.json \
    v_1/src/world_models/activations/${M}/mesopotamian_place/metadata.json
echo "=== Done ${M}. $(date) ==="
