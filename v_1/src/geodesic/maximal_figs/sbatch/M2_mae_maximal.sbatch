#!/bin/bash
#SBATCH --job-name=maxfig_M2
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M2_%j.out

# ===========================================================================
# M2 — per-ruler MAE plots (permodel_mae_ruler, bars_mae_ruler) for the 8-model
#      set, balanced + maximal. RUN AFTER M1 (reads best maximal layers from
#      M1's PLS summaries via --layers-json).
#
# Uses FIXED k=3 (--pls-k 3): honest, no per-draw best-k selection (that path
# was selection-biased / optimistic). Drop --pls-k 3 to restore best-k-per-draw.
# ===========================================================================
set -uo pipefail
echo "=== M2 maximal per-ruler MAE ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MF=v_1/src/geodesic/maximal_figs
EO=v_1/src/geodesic/fig1_followups/error_overlap
OUT=$MF/predictions
mkdir -p "$OUT" "$MF/figures" "$MF/logs"

python -u "$EO/dump_oof_predictions_balanced.py" \
    --out-dir "$OUT" \
    --models tfidf,mlm,thalesian_akk300m,thalesian_cunei400m,qwen3_1b7,qwen3_8b,qwen3_32b,random \
    --cleaning maximal \
    --pls-k 3 \
    --layers-json "$MF/probes" \
    --draw-range 0-199

PRED="$OUT/predictions.csv"
echo ""; echo "--- per-model MAE analysis ---"
python -u "$EO/analyze_per_model.py" --pred-csv "$PRED" --tol 100 --examples 0 \
    || echo "WARN per_model failed"

# collect the two requested plots into the single figures/ folder
for f in permodel_mae_ruler.png bars_mae_ruler.png; do
  [ -f "$OUT/$f" ] && cp -f "$OUT/$f" "$MF/figures/$f" && echo "copied $f -> figures/"
done

echo ""; echo "=== Committing ==="
git add "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.png "$MF/figures"/*.png \
        "$EO/dump_oof_predictions_balanced.py" "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M2: 8-model per-ruler MAE (fixed k=3), balanced+maximal (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed — outputs on cluster fs."
echo "=== Done. $(date) ==="
