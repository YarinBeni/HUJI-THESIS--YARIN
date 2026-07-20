#!/bin/bash
#SBATCH --job-name=J16d_stress_umap
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J16d_stress_umap_%j.out
# ===========================================================================
# J16d — UMAP-2D coords of the stress-test embeddings (maximal + engtier0,
# all models incl mlm+random, L0 + best year layer) — the UMAP counterpart to
# the t-SNE/PCA dump. Feeds the six-view metadata panels + the GUI. Writes the
# committed v_1/src/viz/stress_umap_coords.json (merge-on-write, resumable).
#   sbatch v_1/src/stress_tests/sbatch/J16d_stress_umap.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J16d stress UMAP === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -u v_1/src/stress_tests/eda/dump_stress_umap.py || { echo "FAIL umap"; exit 1; }
commit_push "J16d: stress-test UMAP coords (job ${SLURM_JOB_ID})" \
    v_1/src/viz/stress_umap_coords.json
echo "=== Done. $(date) ==="
