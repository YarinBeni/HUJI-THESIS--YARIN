#!/bin/bash
#SBATCH --job-name=r2_p3_thalesian_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p3_thalesian_mc_%j.out

# ============================================================================
# Round 2 Phase 3 — Balanced MC re-probing for the two Thalesian models.
#
# Uses the SAME draws_matrix.npy that Phase 0 used (200 balanced sub-draws,
# 21 frags/ruler × 8 rulers) so the MC numbers are apples-to-apples comparable
# with Phase 0's tfidf/mlm/qwen/random MC results.
#
# Tier0/mean only (matches what run_mc_probes.py supports out of the box; the
# imbalanced full-grid is covered by the separate probe_thalesian_all.sbatch).
#
# Outputs go to the SAME probes/ dir as Phase 0, so aggregate_p0.py picks them
# up automatically after METHODS list is extended.
# ============================================================================

echo "=== Round 2 Phase 3 — Thalesian balanced MC ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round2_phase0/probes

python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix    v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    --fragment-order  v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    --output-dir      v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes          thalesian_akk300m_pls,thalesian_akk300m_cls,thalesian_cunei400m_pls,thalesian_cunei400m_cls \
    --layers          all \
    || { echo "FAILED: thalesian MC"; exit 1; }

# Auto-push results back to the repo
git add v_1/src/linear_probing/results/orcc_round2_phase0/probes/thalesian_*
git commit -m "Phase 3 balanced MC: Thalesian probes (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo ""
echo "=== Done ==="
echo "End: $(date)"
