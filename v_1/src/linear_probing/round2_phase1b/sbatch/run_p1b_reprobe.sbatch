#!/bin/bash
#SBATCH --job-name=r2_p1b_reprobe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1b_reprobe_%j.out

# ===========================================================================
# Round 2 Phase 1b — Re-probing of prompted activations
#
# Takes the 12 NPZ files written by extract_prompted_acts.py
#   (4 variants pv0/pv1/pv2/pv3 × 3 layers L0/L15/L-1=L28)
# and re-fits the Round-1 CLS and PLS probes on them. The script compares to
# BOTH Round-1 Qwen baselines (last pooling and mean pooling) and emits a
# 4-way verdict (BEATS BOTH / BEATS LAST / BEATS MEAN / FAILS BOTH).
#
# No GPU needed — pure numpy/sklearn. Short walltime.
#
# Output:
#   v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing/
#     {variant}__L{NN}__cls.json   (12 files)
#     {variant}__L{NN}__pls.json   (12 files)
#     phase1b_summary.json
#     phase1b_report.md
# ===========================================================================

echo "=== Round 2 Phase 1b — reprobing ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

# ── Step 1: verify prompted-activation NPZs exist ────────────────────────────
echo ""
echo "--- Step 1: verify activations exist ---"
for v in pv0 pv1 pv2 pv3; do
    d="v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations/$v"
    if [ ! -d "$d" ]; then
        echo "FAILED: missing activations dir for $v: $d"
        exit 1
    fi
    n_npz=$(ls "$d"/*.npz 2>/dev/null | wc -l)
    echo "  $v: $n_npz npz files"
done

# ── Step 2: run reprobe ──────────────────────────────────────────────────────
echo ""
echo "--- Step 2: reprobe_pv.py ---"
python -u v_1/src/linear_probing/round2_phase1b/reprobe_pv.py \
    --acts_root v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations \
    --corpus    v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --out_dir   v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing \
    --variants pv0,pv1,pv2,pv3 \
    --poolings last,mean \
    --layers 0,4,10,15,22,28 \
    || { echo "FAILED: reprobe_pv.py"; exit 1; }

# ── Step 3: push results to GitHub ───────────────────────────────────────────
echo ""
echo "=== Pushing Phase 1b reprobing results to GitHub ==="
git add v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing/
git commit -m "Phase 1b reprobing v2: CLS+PLS x 4 variants x 2 poolings x 6 layers" || true
git push origin main || true

echo ""
echo "=== Done ==="
echo "End: $(date)"
