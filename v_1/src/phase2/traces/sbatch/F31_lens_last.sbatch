#!/bin/bash
#SBATCH --job-name=F31_last
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/phase2/traces/logs/F31_last_%A_%a.out
# ===========================================================================
# F31 — the pooling control on the lens. F6/F21/F29 read only MEAN-pooled
# document directions, which is the unfavourable case: the average of a
# whole inscription is not a next-token state, so junk there is partly
# expected. This re-runs the same instrument (raw lens + tuned lens +
# decile spectroscopy vs 50 translator-matched random directions) on the
# LAST-TOKEN document directions that F19 fitted, alongside the mean ones.
# Reuses the cached translators from F29 (--steps 0 would skip training;
# left on so any new layer gets one).
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b qwen3_8b llama2_7b)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== F31 ${METHOD} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/traces/logs

python -u v_1/src/phase2/traces/lens_tuned.py --method "${METHOD}" \
    || { echo "FAILED F31 ${METHOD}"; exit 1; }

commit_push "F31: lens at both pooling sites ${METHOD} (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/traces/results
echo "=== Done. $(date) ==="
