#!/bin/bash
#SBATCH --job-name=r3_phase_b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_b/logs/scan_${METHOD}_${CLEANING}_${POOL}_%j.log
# Parametrized via --export=METHOD=...,CLEANING=...,POOL=...
# Submit via submit_all.sh, not directly.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/geodesic/results/phase_b/logs

echo "=== Phase B scan: METHOD=${METHOD}  CLEANING=${CLEANING}  POOL=${POOL} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date)"

python -u v_1/src/geodesic/phase_b/scan.py \
    --method   "$METHOD"   \
    --cleaning "$CLEANING" \
    --pool     "$POOL"     \
    --output-dir v_1/src/geodesic/results/phase_b \
    || { echo "FAILED"; exit 1; }

git add v_1/src/geodesic/results/phase_b/phase_b_${METHOD}_${CLEANING}_${POOL}.json || true
git commit -m "Phase B scan: ${METHOD} ${CLEANING} ${POOL} (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
