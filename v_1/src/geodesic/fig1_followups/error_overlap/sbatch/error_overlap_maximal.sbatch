#!/bin/bash
#SBATCH --job-name=fig1_err_max
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/error_overlap/logs/overlap_max_%j.out

# ===========================================================================
# Maximal-cleaning error analysis (the length-equalized control). `maximal`
# truncates every fragment to <=32 words, so it tests whether our tier0
# findings reduce to "longer text = easier". mlm has no maximal activations
# (tier0-only) and skips; runs on thalesian (L9) / qwen3_32b (L7) / tfidf.
# CPU-only probing. Writes to predictions_maximal/ and pushes.
# ===========================================================================

set -uo pipefail
echo "=== Fig-1 maximal error analysis ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EO=v_1/src/geodesic/fig1_followups/error_overlap
OUT=$EO/predictions_maximal
mkdir -p "$EO/logs" "$OUT"

python -u "$EO/dump_oof_predictions.py" --out-dir "$OUT" --cleaning maximal --k 5 --n-splits 5

PRED="$OUT/predictions.csv"
echo ""; echo "--- analyses on maximal ---"
python -u "$EO/error_map.py"        --pred-csv "$PRED" || echo "WARN error_map failed"
python -u "$EO/analyze_per_model.py" --pred-csv "$PRED" --tol 100 --examples 0 || echo "WARN per_model failed"
python -u "$EO/compare_anchor.py"    --pred-csv "$PRED" --anchor tfidf --tol 100 || echo "WARN compare failed"

echo ""; echo "=== Committing ==="
git add "$OUT"/*.csv "$OUT"/*.png
git commit -m "Maximal-cleaning error analysis (length-equalized control, job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed — outputs on cluster fs."
echo ""; echo "=== Done. End: $(date) ==="
