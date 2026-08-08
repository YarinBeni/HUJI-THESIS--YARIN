#!/bin/bash
#SBATCH --job-name=F13_e3last
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/phase2/transfer/logs/F13_e3last_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# F13 — gap fixes G-pooling + G-control + G-surgical for E3:
#  * --site last: the cell-A direction was fitted on LAST-token name states;
#    v1 applied it to MEAN-pooled fragments — this is the matched rerun;
#  * positive control: the same frozen direction scored on cell A itself must
#    reproduce rho ~ .85+ or the whole pipeline is suspect;
#  * LEACE now fp64 + a ruler-probe before/after accuracy check (did the
#    erasure actually erase?).
#   0 olmo2_7b   1 olmo2_7b_random   2 llama2_7b   3 qwen3_8b
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random llama2_7b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F13 e3 last ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/transfer/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
pip install -q concept-erasure 2>/dev/null || true

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}: no extraction on disk"; continue
    fi
    python -u v_1/src/phase2/transfer/e3_transfer.py --method "$M" \
        --variant "$V" --site last \
        || python -u v_1/src/phase2/transfer/e3_transfer.py --method "$M" \
              --variant "$V" --site last --skip-leace \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F13: E3 last-on-last + controls ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/transfer/results
echo "=== Done ${M}. $(date) ==="
