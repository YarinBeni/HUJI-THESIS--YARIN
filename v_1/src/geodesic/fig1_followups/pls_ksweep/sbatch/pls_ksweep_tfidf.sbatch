#!/bin/bash
#SBATCH --job-name=fig1_pls_tfidf
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/pls_ksweep/logs/ksweep_tfidf_%j.out

# ===========================================================================
# Backfill the TF-IDF k-sweep that the original job dropped (its k=128 hit the
# balanced ceiling and the unguarded TF-IDF loop failed every draw — now fixed
# in run_mc_probes.py). TF-IDF has no layers, so it is a single "layer" (L00/na)
# and ignores --layers. Writes into the SAME probes dir + method-tag so the
# aggregations pick it up alongside the 3 neural models, then regenerates all
# three figures and pushes.
# ===========================================================================

set -euo pipefail
echo "=== Fig-1 Task 4: TF-IDF k-sweep backfill ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FU=v_1/src/geodesic/fig1_followups
SUBSET_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
mkdir -p "$FU/pls_ksweep/logs"

# ── TF-IDF only (no layers; k=128 now skips gracefully via the guard) ────────
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET_DIR/draws_matrix.npy" \
    --fragment-order "$SUBSET_DIR/corpus_fragment_order.json" \
    --output-dir     "$FU/pls_ksweep/probes" \
    --method-tag     mc_balanced_ksweep \
    --probes         tfidf_pls \
    --pls-k          1,2,3,5,8,16,32,64,128 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-8}"

# ── regenerate all three figures now that TF-IDF is present ───────────────────
echo ""; echo "--- regenerating figures ---"
python -u "$FU/pls_ksweep/aggregate_and_plot.py"  --probes-dir "$FU/pls_ksweep/probes"
python -u "$FU/pls_ksweep/best_k_compare.py"      --probes-dir "$FU/pls_ksweep/probes" \
        --tradeoff-csv "$FU/pls_ksweep/pls_components_tradeoff.csv" --fixed-k 3
python -u "$FU/pls_ksweep/per_method_panels.py" \
        --tradeoff-csv "$FU/pls_ksweep/pls_components_tradeoff.csv" \
        --bestk-csv    "$FU/pls_ksweep/best_k_vs_fixed_k.csv"

echo ""; echo "=== Committing ==="
git add "$FU/pls_ksweep"/*.png "$FU/pls_ksweep"/*.csv
git commit -m "TF-IDF k-sweep backfill + regenerated figures (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN: pull --rebase failed"
git push origin main || echo "WARN: push failed — outputs on cluster fs."
echo ""; echo "=== Done. End: $(date) ==="
