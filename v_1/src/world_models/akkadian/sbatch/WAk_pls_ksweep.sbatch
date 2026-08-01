#!/bin/bash
#SBATCH --job-name=WAk_pls_ksweep
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAk_%A_%a.out
#SBATCH --array=0-14
# ===========================================================================
# WAk — widen the PLS latent-dimension grid on the fragment cells and de-bias the
# choice of k.
#
# WHY. The thesis deck inherited PLS_KS = {1,2,3,5} from shared/mc_probe.py. That was
# never fitted; it was a default. In the WAgrp results 18 of 58 fragment cells came
# back with pls_best_k == 5 — pinned at the ceiling — which means the grid, not the
# data, was picking k. This job re-runs mc_group with k spanning 1..64 log-spaced.
#
# It also adds a NESTED estimate: k chosen on an inner GroupKFold over the training
# rulers only, then refit and scored on the outer test fold. `pls_best_k` /
# `pls_spearman_mean` are selected on the outer test folds (that is what the deck
# does, so they stay for comparability), but that selection is optimistic and gets
# more so as the grid grows — which is exactly what widening the grid would otherwise
# hide. `pls_nested_spearman_mean` is the honest number to quote.
#
# CPU only — reuses the stored activations, no re-extraction. Overwrites the mc_group
# block written by WAgrp; every other key in the JSON is left alone.
#   sbatch v_1/src/world_models/akkadian/sbatch/WAk_pls_ksweep.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random \
         thalesian_akk300m thalesian_cunei400m umt5_base tfidf)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WAk ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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
    || { echo "FAILED ksweep ${M}"; exit 1; }
commit_push "WAk: PLS k-sweep 1-64 + nested-k mc_group ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes/${M}
echo "=== Done ${M}. $(date) ==="
