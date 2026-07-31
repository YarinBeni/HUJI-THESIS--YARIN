#!/bin/bash
#SBATCH --job-name=WM_figs
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/logs/WM_figs_%A_%a.out
#SBATCH --array=0-13
# ===========================================================================
# WM_figs — the Modell et al. manifold figures computed on the FULL activations
# (not the saved PCA-8 shortcut): L2-normalise -> rank-r uncentered SVD -> kNN graph ->
# Dijkstra -> cosine, then render the arc plot and the two isometry diagnostics with
# Chatterjee xi / Pearson rho boxed, exactly as the paper draws them. Also saves the
# underlying arrays so styling can be reworked offline. CPU, reuses stored acts.
#   sbatch v_1/src/world_models/sbatch/WM_figs.sbatch
#   sbatch --export=ALL,SURFACE=ent --array=0-10 ...   # one surface only
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random \
         thalesian_akk300m thalesian_cunei400m umt5_base)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WM_figs ${M} [${SURFACE:-all}] === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs v_1/src/world_models/manifold/figs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/manifold/manifold_figs.py \
    --method "$M" --surface "${SURFACE:-all}" --knn "${KNN:-10}" --rank "${RANK:-4}" \
    || { echo "FAILED WM_figs ${M}"; exit 1; }
commit_push "WM_figs[${SURFACE:-all}]: full-activation manifold figures ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/manifold/figs
echo "=== Done ${M}. $(date) ==="
