#!/bin/bash
#SBATCH --job-name=W2_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/W2_probe_%A_%a.out
#SBATCH --array=0-13
# ===========================================================================
# W2 — per-layer ridge probes (CPU). Run after the matching W1* extraction landed.
# CLEANUP=1 arms delete their npz after all probes succeed (random arms: no
# round-2 analysis planned for them; trained arms keep acts for PCA/holdouts).
#   0 qwen3_1b7        1 qwen3_8b          2 qwen3_32b         3 gpt_oss_120b
#   4 thal_akk300m     5 thal_cunei400m    6 umt5_base         7 random*
#   8 llama2_7b        9 llama2_13b       10 llama2_70b
#  11 llama2_7b_random* 12 llama2_13b_random* 13 llama2_70b_random*
#   sbatch v_1/src/world_models/sbatch/W2_probe.sbatch
#   sbatch --array=10 v_1/src/world_models/sbatch/W2_probe.sbatch   # one arm
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m \
         thalesian_cunei400m umt5_base random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random)
CLEANUPS=(0 0 0 0 0 0 0 1 0 0 0 1 1 1)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
C="${CLEANUPS[$SLURM_ARRAY_TASK_ID]}"

echo "=== W2 probe ${M} (cleanup=${C}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

EXTRA=""
[ "$C" = "1" ] && EXTRA="--cleanup"
python -u v_1/src/world_models/probe_wm.py --method "$M" $EXTRA \
    || { echo "FAILED probe ${M}"; exit 1; }

commit_push "W2: world-models probes ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/projections/${M}
echo "=== Done ${M}. $(date) ==="
