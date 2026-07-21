#!/bin/bash
#SBATCH --job-name=W1b_extract_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/W1b_extract_gptoss_%j.out
# ===========================================================================
# W1b — gpt-oss-120b extraction (bf16 ~240GB; gpu:8 — J12b OOM'd on gpu:4 during
# weight placement, don't retry smaller). Same protocol as W1.
#   sbatch v_1/src/world_models/sbatch/W1b_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== W1b extract ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/extract_acts.py --method "$M" --batch-size 32 \
    || { echo "FAILED extract ${M}"; exit 1; }

commit_push "W1b: world-models acts ${M} (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
