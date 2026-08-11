#!/bin/bash
#SBATCH --job-name=E3b_wb
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=v_1/src/phase2/transfer/logs/E3b_wb_%A_%a.out
# ===========================================================================
# E3b — the reviewer question: transfer the CELL-B ruler axis (w_B), not just
# the cell-A famous-figure axis. Fits w_B once (canonical split, best layer
# from probe_entity results, cached npz), applies it frozen to fragments,
# LEACE mediation, cosine vs the E1 document direction. --stability 48 refits
# the direction on 48 resampled train sets and reports the spread of every
# read-out — the "which w did you take" answer, as data. The same stability
# pass is run for the cell-A axis.
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b qwen3_8b llama2_7b olmo2_7b_random)
VARS=(eng_tier0 akk_maximal)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 2))]}
VARIANT=${VARS[$((SLURM_ARRAY_TASK_ID % 2))]}
echo "=== E3b ${METHOD}/${VARIANT} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/transfer/logs

python -u v_1/src/phase2/transfer/e3_transfer.py \
    --method "${METHOD}" --variant "${VARIANT}" \
    --entity-set assyrian_ruler --stability 48 \
    || { echo "FAILED E3b ruler ${METHOD}/${VARIANT}"; exit 1; }

python -u v_1/src/phase2/transfer/e3_transfer.py \
    --method "${METHOD}" --variant "${VARIANT}" \
    --entity-set historical_figure --stability 48 \
    || { echo "FAILED E3 stability ${METHOD}/${VARIANT}"; exit 1; }

commit_push "E3b: ruler-axis transfer + w-stability ${METHOD}/${VARIANT} (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/transfer/results
echo "=== Done. $(date) ==="
