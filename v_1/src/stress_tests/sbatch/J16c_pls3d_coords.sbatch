#!/bin/bash
#SBATCH --job-name=J16c_pls3d_coords
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J16c_pls3d_coords_%j.out
# ===========================================================================
# J16c — SUPERVISED PLS k=3 3-D coords of the stress-test embeddings (maximal
# + engtier0, all models incl. mlm+random, best year layer) for 3-D year-
# separability maps. Writes committed v_1/src/viz/pls3d_coords.json
# (merge-on-write, resumable). 3-D scoring + plots happen off-cluster.
#   sbatch v_1/src/stress_tests/sbatch/J16c_pls3d_coords.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J16c PLS-3D coords === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -u v_1/src/stress_tests/eda/dump_pls3d_coords.py || { echo "FAIL pls3d"; exit 1; }
commit_push "J16c: PLS-3D coords for 3-D separability maps (job ${SLURM_JOB_ID})" \
    v_1/src/viz/pls3d_coords.json
echo "=== Done. $(date) ==="
