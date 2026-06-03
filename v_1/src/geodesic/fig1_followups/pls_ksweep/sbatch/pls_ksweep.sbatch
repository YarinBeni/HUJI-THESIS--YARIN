#!/bin/bash
#SBATCH --job-name=fig1_pls_ksweep
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/pls_ksweep/logs/ksweep_%j.out

# ===========================================================================
# Fig-1 follow-up Task 4 — PLS components tradeoff (balanced, 200 draws).
#
# Reuses run_mc_probes.py (the tested balanced-MC machinery) with the new
# --pls-k flag to sweep a wide n_components grid for the 4 Fig-1A models at
# their best layers. Output goes to an ISOLATED dir + method-tag so it never
# collides with the existing `mc_balanced` results (and so resumability does
# not skip them).
#
# CPU-only: activations are already extracted; this is pure probing.
# ===========================================================================

set -euo pipefail
echo "=== Fig-1 Task 4: PLS k-sweep ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SUBSET_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
OUT_DIR=v_1/src/geodesic/fig1_followups/pls_ksweep/probes
mkdir -p "$OUT_DIR" v_1/src/geodesic/fig1_followups/pls_ksweep/logs

# Best balanced layers (from T1): mlm L01, thalesian_cunei400m L12, qwen3_32b L09.
# tfidf has no layers (L00/na) and ignores --layers. Passing the union {1,9,12}
# means each model also computes 2 extra cheap layers we simply don't plot.
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET_DIR/draws_matrix.npy" \
    --fragment-order "$SUBSET_DIR/corpus_fragment_order.json" \
    --output-dir     "$OUT_DIR" \
    --method-tag     mc_balanced_ksweep \
    --probes         tfidf_pls,mlm_pls,thalesian_cunei400m_pls,qwen3_32b_pls \
    --layers         1,9,12 \
    --cleaning       tier0 \
    --pooling        mean \
    --pls-k          1,2,3,5,8,16,32,64,128 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-16}"

echo "=== Done. End: $(date) ==="
echo "Per-draw JSONs in $OUT_DIR (tag mc_balanced_ksweep)."
echo "Next: pull the dir locally, then run aggregate_and_plot.py"
