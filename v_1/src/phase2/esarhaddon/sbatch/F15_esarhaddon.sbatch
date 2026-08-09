#!/bin/bash
#SBATCH --job-name=F15_esarh
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/esarhaddon/logs/F15_esarh_%A_%a.out
#SBATCH --array=0-4
# ===========================================================================
# F15 — E6 Esarhaddon micro-study: year probe + within-ruler pairwise inside
# the only ruler with internal year variance (176 frags, 11 years). Any signal
# here is identity-free by construction. Detection floor stated in the script.
#   0 tfidf_char   1 olmo2_7b   2 olmo2_7b_random   3 qwen3_8b   4 llama2_7b
# ===========================================================================
set -uo pipefail
METHODS=(tfidf_char olmo2_7b olmo2_7b_random qwen3_8b llama2_7b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F15 esarhaddon ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/esarhaddon/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ "$M" != "tfidf_char" ] && \
       [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}"; continue
    fi
    python -u v_1/src/phase2/esarhaddon/e6_micro.py --method "$M" --variant "$V" \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F15: Esarhaddon micro ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/esarhaddon/results
echo "=== Done ${M}. $(date) ==="
