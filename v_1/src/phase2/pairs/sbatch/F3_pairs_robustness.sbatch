#!/bin/bash
#SBATCH --job-name=F3_pairs_rob
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F3_pairs_rob_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# F3 — E1 robustness pass: the written-in --m 100 --draws 10 setting, which
# pushes per-draw coverage of the big ruler-pair grids ~5x further while the
# 1/m_ij weights keep every ruler-pair's total loss contribution equal. If the
# headline m=21 result moves materially here, the balancing choice mattered
# and both numbers get reported; if it doesn't, m=21 stands unqualified.
#
# Headline arms only (the floor plus the arms the deck story leans on):
#   0 tfidf_char   1 olmo2_7b   2 llama2_70b   3 qwen3_8b
#
# Results land in *.m100.json — a separate filename, never overwriting the
# headline run.
#
#   sbatch v_1/src/phase2/pairs/sbatch/F3_pairs_robustness.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(tfidf_char olmo2_7b llama2_70b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F3 robustness ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ "$M" != "tfidf_char" ] && \
       [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}: no extraction on disk"
        continue
    fi
    python -u v_1/src/phase2/pairs/probe_pairs.py --method "$M" --variant "$V" \
        --m 100 --draws 10 --draws-select 3 \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

N=$(ls v_1/src/phase2/pairs/results/probes/${M}.*.m100.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} m100 result files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no m100 results"; exit 1; }

commit_push "F3: pairwise robustness m=100 ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/pairs/results/probes \
    v_1/src/phase2/pairs/results/directions
echo "=== Done ${M}. $(date) ==="
