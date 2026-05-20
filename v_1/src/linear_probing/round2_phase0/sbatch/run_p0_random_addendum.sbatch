#!/bin/bash
#SBATCH --job-name=r2_p0_random
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p0_random_%j.out

# ===========================================================================
# Round 2 Phase 0 — ADDENDUM: random_pls + random_cls only
#
# The main P0 job (Slurm 8056) already produced the 6 non-random probe arms
# on the locked balanced subset. This addendum fills in the 2 missing random
# arms so the 4-method gate (tfidf / mlm / qwen / random) can be evaluated.
#
# Wrapper is resumable per-draw: any (probe, draw) whose JSON already exists
# is skipped — so this run touches ONLY the 400 missing files
# (200 draws x {random_pls, random_cls}). See run_mc_probes.py:42-46.
#
# IMPORTANT: do NOT re-run build_balanced_subset.py — that would shuffle the
# locked draws and invalidate the existing 1200 per-draw JSONs. We only
# verify the artifacts are still in place.
#
# Output additions:
#   v_1/src/linear_probing/results/orcc_round2_phase0/probes/
#     random_{pls,cls}__mc_balanced__draw{000..199}.json   (400 new files)
#     random_{pls,cls}__mc_balanced__summary.json          (2 new summaries)
# ===========================================================================

echo "=== Round 2 Phase 0 — random_{pls,cls} addendum ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

# ── Step 1: verify locked balanced subset is still on disk ──────────────────
echo ""
echo "--- Step 1: verify balanced_subset artifacts (NO rebuild) ---"
SUBSET_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
for f in manifest.json draws_matrix.npy corpus_fragment_order.json; do
    if [ ! -f "$SUBSET_DIR/$f" ]; then
        echo "FAILED: expected artifact missing: $SUBSET_DIR/$f"
        echo "Refusing to proceed — the locked subset must already exist."
        exit 1
    fi
done
echo "Balanced subset artifacts present — OK."

# Sanity: count existing per-draw JSONs for the 6 existing probes (expect 1200).
PROBES_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/probes
EXISTING=$(ls "$PROBES_DIR"/{tfidf,mlm,qwen}_{pls,cls}__mc_balanced__draw*.json 2>/dev/null | wc -l)
echo "Existing per-draw JSONs from prior run: $EXISTING (expect 1200)"

# ── Step 2: run ONLY the two random probes (resumable wrapper) ──────────────
echo ""
echo "--- Step 2: run_mc_probes.py --probes random_pls,random_cls ---"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix "$SUBSET_DIR/draws_matrix.npy" \
    --fragment-order "$SUBSET_DIR/corpus_fragment_order.json" \
    --method-tag mc_balanced \
    --output-dir "$PROBES_DIR" \
    --probes random_pls,random_cls \
    || { echo "FAILED: run_mc_probes.py"; exit 1; }

# Sanity: final summary count (expect 8 total across all 4 methods x 2 pools).
SUMMARIES=$(ls "$PROBES_DIR"/*__mc_balanced__summary.json 2>/dev/null | wc -l)
RANDOM_DRAWS=$(ls "$PROBES_DIR"/random_{pls,cls}__mc_balanced__draw*.json 2>/dev/null | wc -l)
echo ""
echo "Post-run counts:  summaries=$SUMMARIES (expect 8)  random_draws=$RANDOM_DRAWS (expect 400)"

# ── Step 3: push addendum results to GitHub ─────────────────────────────────
echo ""
echo "=== Pushing Phase 0 random addendum to GitHub ==="
git add v_1/src/linear_probing/results/orcc_round2_phase0/probes/
git commit -m "Add Round 2 Phase 0 random addendum: random_{pls,cls} probes (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally."

echo ""
echo "=== Done ==="
echo "End: $(date)"
