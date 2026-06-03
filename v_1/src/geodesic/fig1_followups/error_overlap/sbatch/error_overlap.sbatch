#!/bin/bash
#SBATCH --job-name=fig1_err_overlap
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/error_overlap/logs/overlap_%j.out

# ===========================================================================
# Fig-1 follow-up Task 3 — dump per-fragment OOF year predictions for the 4
# Fig-1A models (imbalanced, GroupKFold-by-ruler, full labeled corpus).
# CPU-only probing; no GPU, no new extraction. Pull predictions/ locally and
# run analyze_overlap.py for the figures.
# ===========================================================================

set -euo pipefail
echo "=== Fig-1 Task 3: error-overlap OOF dump ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p v_1/src/geodesic/fig1_followups/error_overlap/logs

python -u v_1/src/geodesic/fig1_followups/error_overlap/dump_oof_predictions.py \
    --out-dir v_1/src/geodesic/fig1_followups/error_overlap/predictions \
    --k 5 --n-splits 5

echo "=== Done. End: $(date) ==="
echo "Pull predictions/ locally, then: python analyze_overlap.py --pred-csv .../predictions.csv --tol 100"
