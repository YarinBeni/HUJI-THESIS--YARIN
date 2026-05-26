#!/bin/bash
#SBATCH --job-name=r3_phase_c
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_c/logs/loro_${METHOD}_${CLEANING}_${POOL}_L${LAYER}_${RULER_SLUG}_%j.log
# Parametrized via --export=METHOD=...,CLEANING=...,POOL=...,LAYER=...,RULER=...,RULER_SLUG=...

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/geodesic/results/phase_c/logs

echo "=== Phase C LORO: METHOD=${METHOD} CLEANING=${CLEANING} POOL=${POOL} L${LAYER} ruler='${RULER}' ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

python -u v_1/src/geodesic/phase_c/loro.py \
    --method   "$METHOD"   \
    --cleaning "$CLEANING" \
    --pool     "$POOL"     \
    --layer    "$LAYER"    \
    --ruler    "$RULER"    \
    --output-dir v_1/src/geodesic/results/phase_c \
    || { echo "FAILED"; exit 1; }

RULER_SLUG_SAFE=$(echo "$RULER" | tr ' ' '_' | tr '-' '_')
git add "v_1/src/geodesic/results/phase_c/loro_${METHOD}_${CLEANING}_${POOL}_L$(printf '%02d' $LAYER)_${RULER_SLUG_SAFE}.json" || true
git commit -m "Phase C LORO: ${METHOD} ${CLEANING} ${POOL} L${LAYER} ${RULER} (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
