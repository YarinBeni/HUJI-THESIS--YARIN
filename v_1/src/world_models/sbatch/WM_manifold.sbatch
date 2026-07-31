#!/bin/bash
#SBATCH --job-name=WM_manifold
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/logs/WM_manifold_%A_%a.out
#SBATCH --array=0-13
# ===========================================================================
# WM_manifold — 3-phase representation-manifold analysis over the STORED activations
# (CPU, no re-extraction). Per arm it covers both surfaces:
#   fragment level  akk_maximal | eng_tier0  x  year | geo  x  last | mean
#   entity   level  the paper's six English datasets x last | mean
# Phase 1 PCA/UMAP embeddings · Phase 2 isometry (Chatterjee xi, Pearson rho, rho-vs-layer)
# Phase 3 reducibility (mixture + separability indices, spectra).
# Writes small JSON + coords npz so all FIGURES regenerate offline.
#   sbatch v_1/src/world_models/sbatch/WM_manifold.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random \
         thalesian_akk300m thalesian_cunei400m umt5_base)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WM_manifold ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import umap" 2>/dev/null || pip install -q umap-learn
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs v_1/src/world_models/manifold/results
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/manifold/run_manifold.py --method "$M" --surface both \
    || { echo "FAILED manifold ${M}"; exit 1; }
commit_push "WM_manifold: 3-phase manifold analysis ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/manifold/results
echo "=== Done ${M}. $(date) ==="
