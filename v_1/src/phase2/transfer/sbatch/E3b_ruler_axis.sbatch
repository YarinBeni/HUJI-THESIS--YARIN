#!/bin/bash
#SBATCH --job-name=E3b_wb
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=v_1/src/phase2/transfer/logs/E3b_wb_%A_%a.out
# ===========================================================================
# E3b — transfer the CELL-B ruler axis (w_B). One task per METHOD (both
# variants run sequentially inside it) so the ruler-activation extraction —
# which run 25000/25002 showed was cleaned from disk — happens exactly once
# per model, on this task's GPU, before the CPU-side fit + frozen transfer.
# The direction is one canonical fit on the fixed 80/20 split, same
# convention as cell A.
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b qwen3_8b llama2_7b olmo2_7b_random)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== E3b ${METHOD} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/transfer/logs

ACTS="v_1/src/world_models/akkadian/activations/${METHOD}/assyrian_ruler"
if ! ls "${ACTS}"/ent_last.layer*.npz >/dev/null 2>&1; then
    echo "--- ruler activations missing; re-extracting (${METHOD}) ---"
    python -u v_1/src/world_models/akkadian/extract_entity.py \
        --method "${METHOD}" --entity-type assyrian_ruler \
        || { echo "FAILED extract_entity ${METHOD}"; exit 1; }
fi

for VARIANT in eng_tier0 akk_maximal; do
    python -u v_1/src/phase2/transfer/e3_transfer.py \
        --method "${METHOD}" --variant "${VARIANT}" \
        --entity-set assyrian_ruler \
        || { echo "FAILED E3b ruler ${METHOD}/${VARIANT}"; exit 1; }
done

commit_push "E3b: ruler-axis transfer ${METHOD} both variants (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/transfer/results
echo "=== Done. $(date) ==="
