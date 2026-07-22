#!/bin/bash
#SBATCH --job-name=P10_run
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/stress_tests/p10_reduce_kernels/logs/P10_run_%A_%a.out
#SBATCH --array=0-8
# ===========================================================================
# P10 — reduce-then-{gkpls/rbfkpls/krr/dial} on the stored mean acts (CPU).
# One array task per method; sweeps {raw,pca,pls,umap}×{none,zscore,l2} at the
# P9-best layer, maximal + engtier0 cleanings, balanced-MC.
#   sbatch v_1/src/stress_tests/p10_reduce_kernels/sbatch/P10_run.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         thalesian_akk300m thalesian_cunei400m umt5_base mlm)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== P10 run ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import umap" 2>/dev/null || pip install -q umap-learn
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/p10_reduce_kernels/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p10_reduce_kernels/run_acts.py \
    --method "$M" --n-draws 100 --n-jobs 12 \
    || { echo "FAILED P10 ${M}"; exit 1; }

commit_push "P10: reduce-then-kernel ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/stress_tests/p10_reduce_kernels/results/p10__${M}.json
echo "=== Done ${M}. $(date) ==="
