#!/bin/bash
#SBATCH --job-name=r3_phase0_inventory
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=v_1/src/geodesic/results/phase_0_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/projects/lititure-review

git pull origin main || echo "WARNING: git pull failed"

python v_1/src/geodesic/phase_0/inventory.py

git add v_1/src/geodesic/results/phase_0_inventory.json || true
git commit -m "Phase 0: activation inventory (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"
