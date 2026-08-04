#!/bin/bash
#SBATCH --job-name=O5_frag_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/O5_frag_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O5 — probe OLMo's fragment activations under the DECK's protocol. Runs after O4.
#
#   0 olmo2_7b        1 olmo2_7b_random
#
# probe_akk_group.py is the GroupKFold-by-ruler probe (mc_group), not the
# StratifiedKFold one. That distinction is the whole point: r8 has 17 distinct years
# across 8 rulers, so with a ruler in both train and test the label is nearly the
# ruler's identity and even char n-gram TF-IDF reaches .707 on name-stripped text.
# Anything compared against the deck has to come from mc_group, so this arm is
# probed the same way rather than the easy way.
#
# --n-draws 200 and the widened PLS grid (1..64 with nested k selection) match what
# WAgrp and WAk gave the other fifteen arms; --force overwrites any stale block.
#
#   sbatch --dependency=afterok:<O4 jobid> v_1/src/olmo_frequency/sbatch/O5_probe_frag.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O5 probe ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/akkadian/probe_akk_group.py --method "$M" \
    --n-draws 200 --force \
    || { echo "FAILED mc_group ${M}"; exit 1; }

commit_push "O5: fragment mc_group (GroupKFold-by-ruler) ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes/${M}
echo "=== Done ${M}. $(date) ==="
