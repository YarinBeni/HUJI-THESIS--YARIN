#!/bin/bash
#SBATCH --job-name=WBk_entity_ksweep
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=v_1/src/world_models/logs/WBk_%A_%a.out
#SBATCH --array=0-13
# ===========================================================================
# WBk — re-probe CELL B (obscure entity names) with the widened PLS grid.
#
# WB2 scored a single fixed PLS-5. Same objection as WAk: k=5 was an inherited
# default, not a fitted choice. This re-runs probe_entity.py, which now sweeps
# k in {1,2,3,5,8,12,16,24,32,48,64} and records the whole curve as `pls_per_k`
# plus `pls_best_k` / `pls_best_mc`. `pls5_mc` is still written under its old key,
# so nothing already published off these files changes.
#
# CLEANUP IS OFF for every arm — unlike WB2, which deleted the random arms' npz and
# forced the WB5/WB5d re-extraction. Arms whose activations are already gone will log
# "[skip] no ... activations" and exit cleanly; re-extract with WB1*/WB5* first if you
# need them.
#   sbatch v_1/src/world_models/akkadian/sbatch/WBk_entity_ksweep.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m \
         thalesian_cunei400m umt5_base random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== WBk ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/akkadian/probe_entity.py --method "$M" \
    || { echo "FAILED entity ksweep ${M}"; exit 1; }

commit_push "WBk: cell-B entity PLS k-sweep 1-64 ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes_entity/${M}
echo "=== Done ${M}. $(date) ==="
