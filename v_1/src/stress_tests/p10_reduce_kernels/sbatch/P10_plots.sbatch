#!/bin/bash
#SBATCH --job-name=P10_plots
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/stress_tests/p10_reduce_kernels/logs/P10_plots_%j.out
# ===========================================================================
# P10 — 3D reduction figures (PCA/PLS/UMAP/t-SNE × year/ruler × raw/z-score) for a
# few representative arms + TF-IDF. CPU. Run any time.
#   sbatch v_1/src/stress_tests/p10_reduce_kernels/sbatch/P10_plots.sbatch
# ===========================================================================
set -uo pipefail
echo "=== P10 plots === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import umap" 2>/dev/null || pip install -q umap-learn
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/p10_reduce_kernels/logs
P=v_1/src/stress_tests/p10_reduce_kernels/plot_reductions.py

python -u $P --tfidf || true
for M in qwen3_32b thalesian_cunei400m random; do
  for CL in maximal engtier0; do
    python -u $P --method "$M" --cleaning "$CL" || true
  done
done

commit_push "P10: reduction figures (job ${SLURM_JOB_ID})" \
    v_1/src/stress_tests/p10_reduce_kernels/results/fig_p10_reductions__*.png
echo "=== Done. $(date) ==="
