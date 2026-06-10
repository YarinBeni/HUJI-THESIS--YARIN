#!/bin/bash
#SBATCH --job-name=maxfig_M5
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M5_%j.out

# ===========================================================================
# M5 — DEFINITIVE mlm fix. M3 extracted mlm maximal activations, but the mlm
# probes were SKIPPED (run_mc_probes resume saw the empty draw files M1 had
# written before activations existed -> done=0 skipped=200). There is no
# --force flag, so we DELETE the stale mlm draw+summary files and recompute.
#
# RUN AFTER M4 (the k-sweep): M5 also re-renders the k-sweep plot, which needs
# M4's 7-model ksweep draws already present.
#
#   1. purge stale mlm files (standard probes dir + ksweep probes dir)
#   2. recompute mlm standard probes (pls/cls_numeric/cls, all 17 layers)
#   3. recompute mlm k-sweep (wide k, all 17 layers) into ksweep_probes
#   4. re-render fig1/2/4              -> now 8 models
#   5. re-run 8-model per-ruler MAE    -> now 8 models (mlm resolves)
#   6. re-render k-sweep tradeoff/panels -> now 8 models
# CPU-only (mlm is 37M).
# ===========================================================================
set -uo pipefail
echo "=== M5 mlm fix ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MF=v_1/src/geodesic/maximal_figs
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
RMP=v_1/src/linear_probing/round2_phase0/run_mc_probes.py
EO=v_1/src/geodesic/fig1_followups/error_overlap

# 1. purge stale (empty) mlm files so the resume logic recomputes them
echo "--- 1. purge stale mlm files ---"
rm -fv "$MF/probes"/mlm_*__mc_balanced_maximal__*.json
rm -fv "$MF/ksweep_probes"/mlm_*__mc_balanced_ksweep_maximal__*.json

# 2. recompute mlm standard probes (activations now exist)
echo "--- 2. mlm standard probes ---"
python -u "$RMP" \
    --draws-matrix "$SUBSET/draws_matrix.npy" --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir "$MF/probes" --method-tag mc_balanced_maximal \
    --probes mlm_pls,mlm_cls_numeric,mlm_cls --cleaning maximal --pooling mean \
    --layers all --pls-k 1,2,3,5 --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
    || echo "WARN mlm standard probes failed"

# 3. recompute mlm k-sweep (wide k, all 17 layers)
echo "--- 3. mlm k-sweep ---"
python -u "$RMP" \
    --draws-matrix "$SUBSET/draws_matrix.npy" --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir "$MF/ksweep_probes" --method-tag mc_balanced_ksweep_maximal \
    --probes mlm_pls --cleaning maximal --pooling mean \
    --layers all --pls-k 1,2,3,5,8,16,32,64,128 --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
    || echo "WARN mlm ksweep failed"

# 4. re-render fig1/2/4 (8 models)
echo "--- 4. re-render fig1/2/4 ---"
python -u "$MF/make_maximal_figs.py" --probes-dir "$MF/probes" --tag mc_balanced_maximal \
    --tables-out "$MF/tables" --fig-out "$MF/figures" || echo "WARN render failed"

# 5. re-run 8-model MAE (mlm now resolves)
echo "--- 5. 8-model MAE ---"
python -u "$EO/dump_oof_predictions_balanced.py" --out-dir "$MF/predictions" \
    --models tfidf,mlm,thalesian_akk300m,thalesian_cunei400m,qwen3_1b7,qwen3_8b,qwen3_32b,random \
    --cleaning maximal --pls-k 3 --layers-json "$MF/probes" --draw-range 0-199 \
    || echo "WARN dump failed"
python -u "$EO/analyze_per_model.py" --pred-csv "$MF/predictions/predictions.csv" --tol 100 --examples 0 \
    || echo "WARN per_model failed"
for f in permodel_mae_ruler.png bars_mae_ruler.png; do
  [ -f "$MF/predictions/$f" ] && cp -f "$MF/predictions/$f" "$MF/figures/$f" && echo "copied $f"
done

# 6. re-render k-sweep plots (8 models)
echo "--- 6. re-render k-sweep ---"
python -u "$MF/maximal_ksweep.py" --std-probes "$MF/probes" --ksweep-probes "$MF/ksweep_probes" \
    --t2-csv "$MF/tables/T2_year_ridge_maximal.csv" --fig-out "$MF/figures" || echo "WARN ksweep render failed"

echo ""; echo "=== Committing ==="
git add "$MF/figures"/*.png "$MF/tables"/*.csv "$MF/predictions"/*.csv "$MF/predictions"/*.json \
        "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M5: recompute mlm (stale-skip fix) -> all 8 models across fig1/2/4 + MAE + k-sweep (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
