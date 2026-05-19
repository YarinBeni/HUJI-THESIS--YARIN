#!/bin/bash
#SBATCH --job-name=r2_p0_probes
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p0_probes_%j.out

# ===========================================================================
# Round 2 Phase 0 — Balanced subset + MC re-run of all 6 probes
#   1. build_balanced_subset.py     (200 draws × 8 rulers × 21 frags)
#   2. run_mc_probes.py             (6 probes × 200 draws = 1200 runs)
#
# Output:  v_1/src/linear_probing/results/orcc_round2_phase0/
#            balanced_subset/{manifest.json, draws_matrix.npy, draws/, ...}
#            probes/{pls,cls}_results_{tfidf,mlm,qwen}__mc_balanced.json
#
# NOTE: this script assumes `run_mc_probes.py` exists at
#       v_1/src/linear_probing/round2_phase0/run_mc_probes.py
#       (loops the 6 existing 05_compute_*.py scripts over each MC draw and
#        writes results tagged `mc_balanced`). If the file is missing the job
#        will fail at Step 2 — that's intentional fail-loud behavior.
# ===========================================================================

echo "=== Round 2 Phase 0 — balanced subset + MC probes ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round2_phase0/probes

# ── Step 1: build balanced subset (200 MC draws) ─────────────────────────────
echo ""
echo "--- Step 1: build_balanced_subset.py ---"
python -u v_1/src/linear_probing/round2_phase0/build_balanced_subset.py \
    --n_draws 200 --k 21 --seed 42 \
    || { echo "FAILED: build_balanced_subset.py"; exit 1; }

# Verify expected artifacts
SUBSET_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
for f in manifest.json draws_matrix.npy corpus_fragment_order.json; do
    if [ ! -f "$SUBSET_DIR/$f" ]; then
        echo "FAILED: expected artifact missing: $SUBSET_DIR/$f"
        exit 1
    fi
done
echo "Balanced subset artifacts OK."

# ── Step 2: MC re-run of all 6 probes ────────────────────────────────────────
echo ""
echo "--- Step 2: run_mc_probes.py (6 probes × 200 draws) ---"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix "$SUBSET_DIR/draws_matrix.npy" \
    --fragment-order "$SUBSET_DIR/corpus_fragment_order.json" \
    --method-tag mc_balanced \
    --output-dir v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes tfidf_pls,tfidf_cls,mlm_pls,mlm_cls,qwen_pls,qwen_cls \
    || { echo "FAILED: run_mc_probes.py"; exit 1; }

# ── Step 3: push results to GitHub ───────────────────────────────────────────
echo ""
echo "=== Pushing Phase 0 results to GitHub ==="
git add \
    v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/manifest.json \
    v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    v_1/src/linear_probing/results/orcc_round2_phase0/probes/
git commit -m "Add Round 2 Phase 0 results: balanced subset + MC probes (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally."

echo ""
echo "=== Done ==="
echo "End: $(date)"
