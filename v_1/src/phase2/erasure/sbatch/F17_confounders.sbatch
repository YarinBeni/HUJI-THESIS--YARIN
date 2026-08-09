#!/bin/bash
#SBATCH --job-name=F17_erase
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/phase2/erasure/logs/F17_erase_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F17 — E4 confounder erasure: LEACE(genre + provenance + length) fitted per
# train fold; does the pairwise separation and the grouped ridge signal
# survive? Includes the genre-probe before/after check.
#   0 tfidf_char   1 olmo2_7b   2 qwen3_8b
# ===========================================================================
set -uo pipefail
METHODS=(tfidf_char olmo2_7b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F17 erasure ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/erasure/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
pip install -q concept-erasure 2>/dev/null || true

FAILED=0
for V in eng_tier0 akk_maximal; do
    if [ "$M" != "tfidf_char" ] && \
       [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}"; continue
    fi
    python -u v_1/src/phase2/erasure/e4_confounders.py --method "$M" --variant "$V" \
        --draws 30 \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F17: confounder erasure ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/erasure/results
echo "=== Done ${M}. $(date) ==="
