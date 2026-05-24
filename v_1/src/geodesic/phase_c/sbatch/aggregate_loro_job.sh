#!/bin/bash
#SBATCH --job-name=r3_pc_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=v_1/src/geodesic/results/phase_c/logs/aggregate_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING"

mkdir -p v_1/src/geodesic/results/phase_c/logs
python -u v_1/src/geodesic/phase_c/aggregate_loro.py

git add v_1/src/geodesic/results/loro_robustness.json || true
git commit -m "Phase C aggregation: LORO robustness (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING"
