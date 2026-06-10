#!/bin/bash
#SBATCH --job-name=maxfig_M3
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M3_%j.out

# ===========================================================================
# M3 — close the mlm gap. The MLM (37M) was only ever extracted at tier0, so it
# was missing from every maximal figure. This job:
#   1. extracts mlm activations over text_maximal -> orcc_round1/activations/mlm_maximal
#   2. probes mlm at maximal (pls/cls_numeric/cls, all layers) into the same
#      maximal_figs/probes dir + tag, so the figures pick mlm up
#   3. re-renders fig1(ACD)/fig2(AB)/fig4(A) — now all 8 models
#   4. re-runs the 8-model per-ruler MAE (mlm now resolves) -> the 2 MAE plots
# CPU-only (MLM is tiny). Supersedes the 7-model MAE from job 9548.
# ===========================================================================
set -uo pipefail
echo "=== M3 mlm maximal ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MF=v_1/src/geodesic/maximal_figs
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
MLM_OUT=v_1/src/linear_probing/results/orcc_round1/activations/mlm_maximal
mkdir -p "$MLM_OUT" "$MF/logs" "$MF/predictions"

# --- 1. extract mlm activations over text_maximal (all 17 layers) -----------
echo ""; echo "--- 1. extract mlm @ maximal ---"
python -u v_1/src/archive/baseline_mlm/03_extract_seal_embeddings_all_layers.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_maximal \
    --output-dir "$MLM_OUT" \
    || { echo "FAILED: mlm maximal extraction"; exit 1; }

# --- 2. probe mlm @ maximal (appends mlm__maximal__ keys to the probes dir) --
echo ""; echo "--- 2. mlm maximal probes ---"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET/draws_matrix.npy" \
    --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir     "$MF/probes" \
    --method-tag     mc_balanced_maximal \
    --probes         mlm_pls,mlm_cls_numeric,mlm_cls \
    --cleaning       maximal \
    --pooling        mean \
    --layers         all \
    --pls-k          1,2,3,5 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-8}" \
    || echo "WARN mlm probe step failed"

# --- 3. re-render fig1/2/4 (now 8 models) -----------------------------------
echo ""; echo "--- 3. re-render fig1/2/4 ---"
python -u "$MF/make_maximal_figs.py" \
    --probes-dir "$MF/probes" --tag mc_balanced_maximal \
    --tables-out "$MF/tables" --fig-out "$MF/figures" \
    || echo "WARN render failed"

# --- 4. re-run 8-model per-ruler MAE (mlm now resolves) ---------------------
echo ""; echo "--- 4. 8-model MAE (fixed k=3) ---"
python -u "$MF/../fig1_followups/error_overlap/dump_oof_predictions_balanced.py" \
    --out-dir "$MF/predictions" \
    --models tfidf,mlm,thalesian_akk300m,thalesian_cunei400m,qwen3_1b7,qwen3_8b,qwen3_32b,random \
    --cleaning maximal --pls-k 3 --layers-json "$MF/probes" --draw-range 0-199 \
    || echo "WARN dump failed"
PRED="$MF/predictions/predictions.csv"
python -u "$MF/../fig1_followups/error_overlap/analyze_per_model.py" \
    --pred-csv "$PRED" --tol 100 --examples 0 || echo "WARN per_model failed"
for f in permodel_mae_ruler.png bars_mae_ruler.png; do
  [ -f "$MF/predictions/$f" ] && cp -f "$MF/predictions/$f" "$MF/figures/$f" && echo "copied $f"
done

echo ""; echo "=== Committing ==="
git add "$MF/figures"/*.png "$MF/tables"/*.csv "$MF/predictions"/*.csv "$MF/predictions"/*.json \
        "$MLM_OUT/metadata.json" "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M3: add mlm @ maximal (extract+probe) -> all 8 models in fig1/2/4 + MAE (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
