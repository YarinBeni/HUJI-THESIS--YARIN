#!/bin/bash
#SBATCH --job-name=F14_str1lens
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F14_str1lens_%A_%a.out
#SBATCH --array=0-5
# ===========================================================================
# F14 — two smaller gap fixes in one array:
#  * tasks 0-2 (G-stride): the F1 layer-selection sweep used stride 2 and may
#    have missed the Llama family's best eng_tier0 layer — rerun with stride 1
#    (the "why is Llama flat on English?" check).
#  * tasks 3-5 (G-lenscal): rerun the logit-lens with 3 random-direction
#    calibration controls, so "lenses to junk" has a baseline.
#   0 llama2_7b   1 llama2_13b   2 llama2_70b
#   3 lens:olmo2_7b   4 lens:llama2_7b   5 lens:qwen3_8b
# ===========================================================================
set -uo pipefail
echo "=== F14 task ${SLURM_ARRAY_TASK_ID} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

if [ "$SLURM_ARRAY_TASK_ID" -le 2 ]; then
    METHODS=(llama2_7b llama2_13b llama2_70b)
    M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
    python -u v_1/src/phase2/pairs/probe_pairs.py --method "$M" \
        --variant eng_tier0 --layer-stride 1 \
        || { echo "FAILED stride1 ${M}"; exit 1; }
    commit_push "F14: eng stride-1 ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
        v_1/src/phase2/pairs/results/probes v_1/src/phase2/pairs/results/directions
else
    METHODS=(_ _ _ olmo2_7b llama2_7b qwen3_8b)
    M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
    python -u v_1/src/phase2/traces/logit_lens.py --method "$M" \
        || { echo "FAILED lens ${M}"; exit 1; }
    commit_push "F14: lens+random-controls ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
        v_1/src/phase2/traces/results
fi
echo "=== Done. $(date) ==="
