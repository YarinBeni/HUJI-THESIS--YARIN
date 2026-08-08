#!/bin/bash
#SBATCH --job-name=F5_e3
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/phase2/transfer/logs/F5_e3_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# F5 — E3 frozen name-direction transfer + LEACE mediation.
#
# Applies each arm's cell-A ridge direction (probe_wm best layer, zero
# document-side fitting) to the fragment activations, before and after LEACE
# erasure of ruler identity, plus the cosine against E1's pairwise direction.
#
#   0 olmo2_7b   1 olmo2_7b_random   2 llama2_7b   3 qwen3_8b
#
# If the cell-A direction npz is missing (gitignored, cluster-local), the job
# regenerates it by re-running probe_wm for that arm first — CPU, walks the
# cell-A npz already on this filesystem.
#
#   sbatch v_1/src/phase2/transfer/sbatch/F5_e3_transfer.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random llama2_7b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F5 e3 ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/transfer/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
pip install -q concept-erasure 2>/dev/null || echo "[warn] pip install failed; --skip-leace fallback below"

if ! ls v_1/src/world_models/results/directions/${M}/historical_figure.*.npz >/dev/null 2>&1; then
    echo "[dir] no saved cell-A direction for ${M} — regenerating via probe_wm"
    python -u v_1/src/world_models/probe_wm.py --method "$M" \
        --entity-type historical_figure \
        || { echo "FAILED probe_wm ${M}"; exit 1; }
fi

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}: no extraction on disk"
        continue
    fi
    python -u v_1/src/phase2/transfer/e3_transfer.py --method "$M" --variant "$V" \
        || python -u v_1/src/phase2/transfer/e3_transfer.py --method "$M" \
              --variant "$V" --skip-leace \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

N=$(ls v_1/src/phase2/transfer/results/${M}.*.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} result files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no results"; exit 1; }

commit_push "F5: E3 transfer ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/transfer/results
echo "=== Done ${M}. $(date) ==="
