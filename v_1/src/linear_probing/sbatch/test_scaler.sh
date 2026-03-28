#!/bin/bash
#SBATCH --job-name=test_scaler
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/test_scaler_%j.out

echo "=== Scaler Speed Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

python v_1/src/linear_probing/test_scaler_speed.py

echo "End: $(date)"
