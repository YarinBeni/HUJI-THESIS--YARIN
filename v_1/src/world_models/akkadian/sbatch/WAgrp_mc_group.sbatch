#!/bin/bash
#SBATCH --job-name=WAgrp_mc_group
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAgrp_%A_%a.out
#SBATCH --array=0-14
# ===========================================================================
# WAgrp — add the DECK'S protocol (GroupKFold-by-ruler inside the 200 balanced draws)
# to the fragment probe results. The existing `mc` block is StratifiedKFold-by-ruler,
# which leaks ruler identity: since r8 `year` has only 17 values over 8 rulers, a pure
# char-n-gram baseline reaches .707 on name-stripped text there, versus .266 in the
# deck. This job writes a `mc_group` block alongside it (nothing is overwritten).
# CPU only — reuses the stored activations, no re-extraction.
#   sbatch v_1/src/world_models/akkadian/sbatch/WAgrp_mc_group.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random \
         thalesian_akk300m thalesian_cunei400m umt5_base tfidf)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WAgrp ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
EXTRA=""; [ "$M" = "tfidf" ] && EXTRA="--tfidf"
[ -z "$EXTRA" ] && EXTRA="--method $M"
python -u v_1/src/world_models/akkadian/probe_akk_group.py $EXTRA --n-draws 200 --force \
    || { echo "FAILED mc_group ${M}"; exit 1; }
commit_push "WAgrp: mc_group (GroupKFold-by-ruler, deck protocol) ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes/${M}
echo "=== Done ${M}. $(date) ==="
