#!/bin/bash
#SBATCH --job-name=WB2_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/logs/WB2_probe_%A_%a.out
#SBATCH --array=0-13
# ===========================================================================
# WB2 — CELL-B entity-level probes (CPU only). Run after the matching WB1*.
# Per arm: 2 datasets x 4 sites x every kept layer, each scored with a 200-draw
# entity-level Monte-Carlo (ridge + PLS-5) plus the fixed hold-out, and again on
# the bare-name rows alone (the exact paper-faithful probe).
#   0 qwen3_1b7        1 qwen3_8b          2 qwen3_32b         3 gpt_oss_120b
#   4 thal_akk300m     5 thal_cunei400m    6 umt5_base         7 random*
#   8 llama2_7b        9 llama2_13b       10 llama2_70b
#  11 llama2_7b_random* 12 llama2_13b_random* 13 llama2_70b_random*
#   (* = --cleanup: delete this arm's entity npz once all its probes succeed)
#   sbatch v_1/src/world_models/akkadian/sbatch/WB2_probe_entity.sbatch
#   sbatch --array=1 v_1/src/world_models/akkadian/sbatch/WB2_probe_entity.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m \
         thalesian_cunei400m umt5_base random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random)
CLEANUPS=(0 0 0 0 0 0 0 1 0 0 0 1 1 1)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
C="${CLEANUPS[$SLURM_ARRAY_TASK_ID]}"

echo "=== WB2 probe ${M} (cleanup=${C}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

EXTRA=""
[ "$C" = "1" ] && EXTRA="--cleanup"
python -u v_1/src/world_models/akkadian/probe_entity.py --method "$M" $EXTRA \
    || { echo "FAILED probe ${M}"; exit 1; }

commit_push "WB2: cell-B entity probes ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes_entity/${M}
echo "=== Done ${M}. $(date) ==="
