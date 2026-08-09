#!/bin/bash
#SBATCH --job-name=F21_spectro
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/phase2/traces/logs/F21_spectro_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F21 — E4.4b lens spectroscopy: the whole-vocabulary upgrade of the logit
# lens. Rank-decile category composition (with per-bucket random-direction
# nulls and z-scores), the year-token order test rho(year value, l_t), and
# the cosine variant that removes the loud-row confound. Strictly more
# sensitive than F14's pole check — the honest instrument for whether the
# document direction carries ANY vocabulary-readable meaning.
#   0 olmo2_7b   1 llama2_7b   2 qwen3_8b
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b llama2_7b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F21 spectroscopy ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/traces/logs

python -u v_1/src/phase2/traces/lens_spectroscopy.py --method "$M" \
    || { echo "FAILED spectroscopy ${M}"; exit 1; }

commit_push "F21: lens spectroscopy ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/traces/results
echo "=== Done ${M}. $(date) ==="
