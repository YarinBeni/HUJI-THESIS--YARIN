#!/bin/bash
#SBATCH --job-name=F4_e8
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F4_e8_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# F4 — E8 inference for E1: dyadic (two-way ruler) bootstrap on the paired
# arm-floor / arm-twin differences, and ruler-level permutation with refit.
# This is what turns the E1 pattern into p-values whose n is 40 rulers,
# not 628k pairs.
#
#   0  eng_tier0   : floor + OLMo-2 + its twin + Qwen3-8B + shared random
#   1  akk_maximal : floor + OLMo-2 + its twin (null expected — the honest check
#                    that the machinery does NOT manufacture significance where
#                    E1 saw none)
#
# Twins are scored (for the contrasts) but skipped in the expensive permutation.
# CPU only; needs the akkadian npz store.
#
#   sbatch v_1/src/phase2/pairs/sbatch/F4_e8_inference.sbatch
# ===========================================================================
set -uo pipefail
echo "=== F4 e8 task ${SLURM_ARRAY_TASK_ID} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u v_1/src/phase2/pairs/e8_inference.py --variant eng_tier0 \
        --arms tfidf_char olmo2_7b olmo2_7b_random qwen3_8b random \
        --skip-perm olmo2_7b_random random \
        --n-perm 150 --perm-draws 2 \
        || { echo "FAILED e8 eng_tier0"; exit 1; }
else
    python -u v_1/src/phase2/pairs/e8_inference.py --variant akk_maximal \
        --arms tfidf_char olmo2_7b olmo2_7b_random \
        --skip-perm olmo2_7b_random \
        --n-perm 150 --perm-draws 2 \
        || { echo "FAILED e8 akk_maximal"; exit 1; }
fi

N=$(ls v_1/src/phase2/pairs/results/inference/*.json 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "FAILED: no inference results"; exit 1; }

commit_push "F4: E8 inference task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/pairs/results/inference
echo "=== Done. $(date) ==="
