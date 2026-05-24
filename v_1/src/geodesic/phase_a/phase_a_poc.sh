#!/bin/bash
#SBATCH --job-name=r3_phase_a_poc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_a_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/projects/HUJI-THESIS--YARIN

git pull origin main || echo "WARNING: git pull failed"

python v_1/src/geodesic/phase_a/poc_thalesian_L12.py

git add v_1/src/geodesic/results/phase_a_results.json || true
git commit -m "Phase A POC results (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"
