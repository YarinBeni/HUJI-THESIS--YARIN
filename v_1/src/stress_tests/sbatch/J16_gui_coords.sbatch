#!/bin/bash
#SBATCH --job-name=J16_gui_coords
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J16_gui_coords_%j.out
# ===========================================================================
# J16 — dump 2-D PCA + t-SNE coords of the maxking-mean embeddings (all models,
# L0 + best maxking layer) and the P3 anchors, for the seal_eda GUI. Writes the
# small committed v_1/src/viz/maxking_coords.json; the GUI merge/standalone
# rebuild happens off-cluster after this lands.
#   sbatch v_1/src/stress_tests/sbatch/J16_gui_coords.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J16 GUI coords === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -u v_1/src/stress_tests/eda/dump_gui_coords.py || { echo "FAIL coords"; exit 1; }
commit_push "J16: maxking GUI coords (job ${SLURM_JOB_ID})" v_1/src/viz/maxking_coords.json
echo "=== Done. $(date) ==="
