#!/bin/bash
#SBATCH --job-name=r3_phase_d
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_d/logs/phase_d_${METHOD}_${CLEANING}_${POOL}_L${LAYER}_%j.log
# Parametrized via --export=METHOD=...,CLEANING=...,POOL=...,LAYER=...

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING"

mkdir -p v_1/src/geodesic/results/phase_d/logs

echo "=== Phase D: ${METHOD} ${CLEANING} ${POOL} L${LAYER} ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

python -u v_1/src/geodesic/phase_d/centroid_spline.py \
    --method   "$METHOD"   \
    --cleaning "$CLEANING" \
    --pool     "$POOL"     \
    --layer    "$LAYER"    \
    --output-dir v_1/src/geodesic/results/phase_d \
    || { echo "FAILED"; exit 1; }

git add v_1/src/geodesic/results/phase_d/ || true
git commit -m "Phase D: centroid+spline ${METHOD} ${CLEANING} ${POOL} L${LAYER} (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING"

echo "=== Done. End: $(date) ==="
