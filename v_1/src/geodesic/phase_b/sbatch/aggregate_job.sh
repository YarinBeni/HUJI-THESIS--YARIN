#!/bin/bash
#SBATCH --job-name=r3_phase_b_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=v_1/src/geodesic/results/phase_b/logs/aggregate_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/geodesic/results/phase_b/logs

echo "=== Phase B aggregation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start : $(date)"

python -u v_1/src/geodesic/phase_b/aggregate.py

git add v_1/src/geodesic/results/geodesic_layer_scoreboard.json \
        v_1/src/geodesic/results/geodesic_best_layers.json || true
git commit -m "Phase B aggregation: geodesic scoreboard + best layers (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
