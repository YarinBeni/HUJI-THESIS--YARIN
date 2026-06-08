#!/bin/bash
#SBATCH --job-name=fig1_err_t0bal
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/error_overlap/logs/overlap_t0bal_%j.out

# ===========================================================================
# BALANCED + tier0 error analysis — the 4th (missing) cell of the
# {imbalanced,balanced} x {tier0,maximal} design. Same per-draw PLS best-k OOF
# pipeline as the balanced+maximal job, but on tier0 (FULL length) text/acts.
# Removes the dominant-class crutch while KEEPING length, so comparing this to
# predictions_maximal_balanced/ isolates the LENGTH effect from the CLASS effect.
# 3 models: tfidf (per-draw fit) + thalesian_cunei400m (L12) + qwen3_32b (L9).
# CPU-only. Writes predictions_tier0_balanced/ then the standard analyses.
# ===========================================================================

set -uo pipefail
echo "=== Fig-1 BALANCED tier0 error analysis ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EO=v_1/src/geodesic/fig1_followups/error_overlap
OUT=$EO/predictions_tier0_balanced
mkdir -p "$EO/logs" "$OUT"

python -u "$EO/dump_oof_predictions_balanced.py" \
    --out-dir "$OUT" \
    --models tfidf,thalesian_cunei400m,qwen3_32b \
    --cleaning tier0 \
    --draw-range 0-199

PRED="$OUT/predictions.csv"
echo ""; echo "--- analyses on balanced+tier0 ---"
python -u "$EO/error_map.py"         --pred-csv "$PRED" || echo "WARN error_map failed"
python -u "$EO/analyze_per_model.py" --pred-csv "$PRED" --tol 100 --examples 0 || echo "WARN per_model failed"
python -u "$EO/compare_anchor.py"    --pred-csv "$PRED" --anchor tfidf --tol 100 || echo "WARN compare failed"
python -u "$EO/balanced_summary.py"  --pred-csv "$PRED" || echo "WARN summary failed"

echo ""; echo "=== Committing ==="
git add "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.png
git commit -m "Balanced+tier0 error analysis: 4th 2x2 cell, isolates length from class (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed — outputs on cluster fs."
echo ""; echo "=== Done. End: $(date) ==="
