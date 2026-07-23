#!/bin/bash
#SBATCH --job-name=Weng_pls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/logs/Weng_pls_%A_%a.out
#SBATCH --array=0-11
# ===========================================================================
# Weng_pls — English (G&T redo) best-layer PLS-k sweep (k up to 64) over the stored
# English acts, so encoders enter the PLS plots. CPU, no re-extraction.
#   0 qwen3_1b7 1 qwen3_8b 2 qwen3_32b 3 thalesian_akk300m 4 thalesian_cunei400m
#   5 umt5_base 6 random 7 llama2_7b 8 llama2_13b 9 llama2_70b
#   10 llama2_7b_random 11 llama2_13b_random
#   sbatch v_1/src/world_models/sbatch/Weng_pls.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base \
         random llama2_7b llama2_13b llama2_70b llama2_7b_random llama2_13b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Weng_pls ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/probe_eng_pls.py --method "$M" \
    || { echo "FAILED eng_pls ${M}"; exit 1; }
commit_push "Weng_pls: English best-layer PLS-k ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/eng_pls/${M}
echo "=== Done ${M}. $(date) ==="
