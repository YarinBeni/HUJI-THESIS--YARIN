#!/bin/bash
#SBATCH --job-name=maxfig_M4
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M4_%j.out

# ===========================================================================
# M4 — MAXIMAL PLS k-sweep (components tradeoff + per-method panels), 8 models.
# RUN AFTER M3 (needs mlm maximal activations + every model's best maximal layer).
#
# Sweeps wide k = 1,2,3,5,8,16,32,64,128 at each model's BEST maximal layer
# (resolved from the M1/M3 standard probe summaries), balanced/maximal/mean.
# Writes ksweep_tradeoff_maximal.png + ksweep_per_method_maximal.png.
# ===========================================================================
set -uo pipefail
echo "=== M4 maximal k-sweep ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MF=v_1/src/geodesic/maximal_figs
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
mkdir -p "$MF/ksweep_probes" "$MF/logs"

# best maximal layer per model -> union, from the standard probe summaries
LAYERS=$(python -u "$MF/maximal_ksweep.py" --emit-layers --std-probes "$MF/probes")
echo "best-layer union: $LAYERS"

PROBES="tfidf_pls,mlm_pls,thalesian_akk300m_pls,thalesian_cunei400m_pls,qwen3_1b7_pls,qwen3_8b_pls,qwen3_32b_pls,random_pls"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET/draws_matrix.npy" \
    --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir     "$MF/ksweep_probes" \
    --method-tag     mc_balanced_ksweep_maximal \
    --probes         "$PROBES" \
    --cleaning       maximal \
    --pooling        mean \
    --layers         "$LAYERS" \
    --pls-k          1,2,3,5,8,16,32,64,128 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-16}" \
    || echo "WARN ksweep probe step failed"

echo ""; echo "--- render maximal k-sweep figures ---"
python -u "$MF/maximal_ksweep.py" \
    --std-probes "$MF/probes" --ksweep-probes "$MF/ksweep_probes" \
    --t2-csv "$MF/tables/T2_year_ridge_maximal.csv" --fig-out "$MF/figures" \
    || echo "WARN render failed"

echo ""; echo "=== Committing ==="
git add "$MF/figures"/ksweep_*.png "$MF/tables"/ksweep_*.csv \
        "$MF/maximal_ksweep.py" "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M4: PLS k-sweep (tradeoff + per-method) maximal/mean/balanced, 8 models (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
