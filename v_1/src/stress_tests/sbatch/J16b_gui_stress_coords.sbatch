#!/bin/bash
#SBATCH --job-name=J16b_gui_stress_coords
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J16b_gui_stress_coords_%j.out
# ===========================================================================
# J16b — dump 2-D PCA + t-SNE coords of the STRESS-TEST embeddings (maximal /
# engtier0 / engmaximal, all models incl. mlm+random, L0 + best year layer)
# for the seal_eda GUI. Writes the committed v_1/src/viz/stress_coords.json
# (merge-on-write, resumable). The GUI merge (05_merge_stress_coords.py) and
# standalone rebuild happen off-cluster after this lands.
#   sbatch v_1/src/stress_tests/sbatch/J16b_gui_stress_coords.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J16b GUI stress coords === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -u v_1/src/stress_tests/eda/dump_stress_coords.py || { echo "FAIL coords"; exit 1; }
commit_push "J16b: stress-test GUI coords — maximal/engtier0/engmaximal (job ${SLURM_JOB_ID})" \
    v_1/src/viz/stress_coords.json
echo "=== Done. $(date) ==="
