#!/bin/bash
#SBATCH --job-name=maxfig_M4
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M4_%j.out

# ===========================================================================
# M4 — MAXIMAL PLS k-sweep (components tradeoff + per-method panels), 9 models.
# RUN AFTER M3 (needs mlm maximal activations + every model's best maximal layer).
# Also requires gpt_oss_120b activations extracted (FT0_extract_gptoss_base.sbatch).
#
# Sweeps wide k = 1,2,3,5,8,16,32,64 at each model's BEST maximal layer
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
mkdir -p "$MF/ksweep_probes" "$MF/logs" "$MF/probes"

# Make gpt_oss_120b probe summary visible to maximal_ksweep.py --emit-layers
# (its probe lives in the finetune results tree, not the standard MF/probes dir)
GPT_PROBE=v_1/src/finetune/results/probes/gpt_oss_120b_pls__mc_balanced_maximal__summary.json
if [ -f "$GPT_PROBE" ]; then
    cp "$GPT_PROBE" "$MF/probes/gpt_oss_120b_pls__mc_balanced_maximal__summary.json"
    echo "Copied gpt_oss_120b probe summary to $MF/probes/"
else
    echo "WARN: gpt_oss_120b probe summary not found at $GPT_PROBE — model will be skipped"
fi

# Purge stale gpt_oss_120b ksweep draws. They were first computed (job 9650)
# before gpt-oss resolved its best layer, so they only cover the OTHER models'
# union layers and MISS gpt-oss's own best layer L5. run_mc_probes resume skips
# a draw whole-file if it exists, so it never back-fills the missing layer ->
# the renderer's collect(L05) finds nothing and gpt-oss is dropped. Deleting
# forces a fresh recompute at the current (gpt-oss-inclusive) layer union.
echo "--- purge stale gpt_oss_120b ksweep draws (missing-layer fix) ---"
rm -fv "$MF/ksweep_probes"/gpt_oss_120b_pls__mc_balanced_ksweep_maximal__*.json

# best maximal layer per model -> union, from the standard probe summaries
LAYERS=$(python -u "$MF/maximal_ksweep.py" --emit-layers --std-probes "$MF/probes")
echo "best-layer union: $LAYERS"

# NOTE: tfidf excluded — its char-n-gram dim is huge, so high-k PLS is pathologically
# slow (k=64 PLS on ~tens-of-thousands of dims x 200 draws blocked the whole job for
# 3h+). TF-IDF is a flat surface baseline already shown as a point in fig1/fig2; its
# k-curve, if wanted, is cheap via the local pls_ksweep/tfidf_ksweep_local.py path.
PROBES="mlm_pls,thalesian_akk300m_pls,thalesian_cunei400m_pls,qwen3_1b7_pls,qwen3_8b_pls,qwen3_32b_pls,random_pls,gpt_oss_120b_pls"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET/draws_matrix.npy" \
    --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir     "$MF/ksweep_probes" \
    --method-tag     mc_balanced_ksweep_maximal \
    --probes         "$PROBES" \
    --cleaning       maximal \
    --pooling        mean \
    --layers         "$LAYERS" \
    --pls-k          1,2,3,5,8,16,32,64 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-32}" \
    || echo "WARN ksweep probe step failed"
# NB: k=128 dropped — balanced sample caps PLS components at 126, so k=128 was
# always skipped (pure wasted attempts). k=64 covers the high end. run_mc_probes
# writes per-draw JSONs and skips completed ones, so a resubmit RESUMES the sweep.

echo ""; echo "--- render maximal k-sweep figures ---"
python -u "$MF/maximal_ksweep.py" \
    --std-probes "$MF/probes" --ksweep-probes "$MF/ksweep_probes" \
    --t2-csv "$MF/tables/T2_year_ridge_maximal.csv" --fig-out "$MF/figures" \
    || echo "WARN render failed"

echo ""; echo "=== Committing ==="
git add "$MF/figures"/ksweep_*.png "$MF/tables"/ksweep_*.csv \
        "$MF/maximal_ksweep.py" "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M4: PLS k-sweep (tradeoff + per-method) maximal/mean/balanced, 9 models incl. gpt_oss_120b (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
