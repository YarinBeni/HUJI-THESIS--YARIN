#!/bin/bash
#SBATCH --job-name=r2_p1b_reprobe_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1b_reprobe_agg_%j.out

# ===========================================================================
# Round 2 Phase 1b — Aggregator (runs AFTER the array job completes)
#
# Submit with: sbatch --dependency=afterok:<ARRAY_JOBID> run_p1b_reprobe_aggregate.sbatch
#
# Reads all 48 per-combo JSONs (4 variants x 2 poolings x 6 layers) produced
# by the array job, then builds:
#   - phase1b_summary.json (with full Round-1 leaderboard baselines)
#   - phase1b_report.md (markdown with the comparison tables and verdict)
#
# Tiny job — no GPU, 15 min walltime is overkill but safe.
# ===========================================================================

echo "=== Round 2 Phase 1b — reprobe aggregator ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

# Sanity: count the per-combo JSONs that should already be on disk.
N_JSONS=$(ls v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing/pv*__*__L*__cls.json 2>/dev/null | wc -l)
echo "Per-combo CLS JSONs on disk: $N_JSONS  (expected 48: 4 variants x 2 poolings x 6 layers)"

# Re-run reprobe_pv.py with ALL variants/poolings/layers. The resumability
# logic loads every per-combo JSON from disk (no probe re-fit) and just
# emits the consolidated summary + report.
python -u v_1/src/linear_probing/round2_phase1b/reprobe_pv.py \
    --acts_root v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations \
    --corpus    v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --out_dir   v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing \
    --variants  pv0,pv1,pv2,pv3 \
    --poolings  last,mean \
    --layers    0,4,10,15,22,28 \
    || { echo "FAILED: aggregator"; exit 1; }

# Push consolidated outputs to GitHub
echo ""
echo "=== Pushing aggregator results to GitHub ==="
git add v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing/
git commit -m "P1b reprobe v3: parallel array + full Round-1 leaderboard" || true
git push origin main || true

echo ""
echo "=== Done aggregator ==="
echo "End: $(date)"
