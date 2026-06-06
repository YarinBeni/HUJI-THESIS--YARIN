#!/bin/bash
#SBATCH --job-name=fig1_err_maxbal
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/error_overlap/logs/overlap_maxbal_%j.out

# ===========================================================================
# BALANCED + maximal error analysis. Re-fits PLS (best k of [1,2,3,5] by OOF
# Spearman) inside each of the 200 Monte-Carlo balanced draws (8 rulers x 21),
# then averages each fragment's out-of-fold predicted year over the draws it
# appears in. This removes BOTH crutches at once: text length (maximal, <=32
# words) and dominant-class weighting (balanced). Covers the ~1076 fragments
# of the 8 well-attested rulers.
#
# Runs all 3 models from one identical pipeline/seed: tfidf (per-draw fit) +
# thalesian_cunei400m (L9) + qwen3_32b (L7), maximal activations. CPU-only.
# Writes predictions_maximal_balanced/ then runs the standard error analyses.
# ===========================================================================

set -uo pipefail
echo "=== Fig-1 BALANCED maximal error analysis ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EO=v_1/src/geodesic/fig1_followups/error_overlap
OUT=$EO/predictions_maximal_balanced
mkdir -p "$EO/logs" "$OUT"

python -u "$EO/dump_oof_predictions_balanced.py" \
    --out-dir "$OUT" \
    --models tfidf,thalesian_cunei400m,qwen3_32b \
    --draw-range 0-199

PRED="$OUT/predictions.csv"
echo ""; echo "--- analyses on balanced+maximal ---"
python -u "$EO/error_map.py"         --pred-csv "$PRED" || echo "WARN error_map failed"
python -u "$EO/analyze_per_model.py" --pred-csv "$PRED" --tol 100 --examples 0 || echo "WARN per_model failed"
python -u "$EO/compare_anchor.py"    --pred-csv "$PRED" --anchor tfidf --tol 100 || echo "WARN compare failed"

echo ""; echo "=== Committing ==="
git add "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.png "$EO/dump_oof_predictions_balanced.py" \
        "$EO/sbatch/error_overlap_maximal_balanced.sbatch"
git commit -m "Balanced+maximal error analysis: per-draw PLS best-k, mean OOF (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed — outputs on cluster fs."
echo ""; echo "=== Done. End: $(date) ==="
