#!/bin/bash
#SBATCH --job-name=F16_seriate
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/phase2/seriation/logs/F16_seriate_%A_%a.out
#SBATCH --array=0-4
# ===========================================================================
# F16 — E7 spectral seriation: the one label-free experiment. Fiedler ordering
# of the fragment kNN graph; labels used once, post hoc, to ask what the
# natural order corresponds to (year? genre? provenance? length?).
# Each task runs both variants + the within-Esarhaddon restriction.
#   0 tfidf_char   1 olmo2_7b   2 olmo2_7b_random   3 qwen3_8b   4 llama2_70b
# ===========================================================================
set -uo pipefail
METHODS=(tfidf_char olmo2_7b olmo2_7b_random qwen3_8b llama2_70b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F16 seriation ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/seriation/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ "$M" != "tfidf_char" ] && \
       [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}"; continue
    fi
    python -u v_1/src/phase2/seriation/e7_seriation.py --method "$M" --variant "$V" \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
    python -u v_1/src/phase2/seriation/e7_seriation.py --method "$M" --variant "$V" \
        --ruler Esarhaddon \
        || { echo "FAILED ${M}/${V}/esarhaddon"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F16: seriation ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/seriation/results
echo "=== Done ${M}. $(date) ==="
