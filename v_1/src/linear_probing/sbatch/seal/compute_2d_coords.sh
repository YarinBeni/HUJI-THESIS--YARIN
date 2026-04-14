#!/bin/bash
#SBATCH --job-name=seal_2d_coords
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/linear_probing/logs/seal_2d_coords_%j.out

echo "=== SEAL 2D Coordinate Computation (t-SNE + PCA) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

python -u v_1/src/linear_probing/04_compute_2d_coords.py \
    || { echo "FAILED: 2D coord computation"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
