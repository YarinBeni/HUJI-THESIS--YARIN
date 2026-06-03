#!/bin/bash
#SBATCH --job-name=fig1_analyze
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/logs/analyze_%j.out

# ===========================================================================
# Fig-1 follow-ups — run BOTH local analyses on the cluster (no interactive
# python available) and push only the small outputs (PNGs + summary CSVs) back
# through git, so Yarin can pull them locally to view.
#
# Submit AFTER the k-sweep (9332) finishes, e.g.:
#     sbatch --dependency=afterok:<ksweep_jobid> \
#       v_1/src/geodesic/fig1_followups/sbatch/analyze_and_push.sbatch
#
# The error-overlap dump (9333) is already done; its predictions.csv just has
# to exist. If the k-sweep probes dir is empty this still runs the error-overlap
# analysis and warns on the missing k-sweep data.
# ===========================================================================

set -uo pipefail
echo "=== Fig-1 analyze + push ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

FU=v_1/src/geodesic/fig1_followups
mkdir -p "$FU/logs"

# ── Task 3 analysis: error overlap + per-label heatmaps ──────────────────────
echo ""; echo "--- analyze_overlap.py (tol=100) ---"
python -u "$FU/error_overlap/analyze_overlap.py" \
    --pred-csv "$FU/error_overlap/predictions/predictions.csv" \
    --tol 100 || echo "WARN: error-overlap analysis failed"

# ── Task 4 aggregation: PLS components tradeoff curve ────────────────────────
echo ""; echo "--- aggregate_and_plot.py (k-sweep) ---"
python -u "$FU/pls_ksweep/aggregate_and_plot.py" \
    --probes-dir "$FU/pls_ksweep/probes" || echo "WARN: k-sweep aggregation failed"

# ── Push only the small artifacts (PNGs + summary CSVs), not raw JSONs ────────
echo ""; echo "=== Committing outputs ==="
git add \
    "$FU/error_overlap/predictions/predictions.csv" \
    "$FU/error_overlap/predictions/overlap_summary.csv" \
    "$FU/error_overlap/predictions"/*.png \
    "$FU/pls_ksweep/pls_components_tradeoff.png" \
    "$FU/pls_ksweep/pls_components_tradeoff.csv" 2>/dev/null

git commit -m "Fig-1 follow-up results: error-overlap + PLS k-sweep figures (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing new to commit"
# commit-first then rebase so we replay cleanly on top of remote (avoid races)
git pull --rebase origin main || echo "WARN: pull --rebase failed"
git push origin main || echo "WARN: push failed — outputs are on the cluster filesystem."

echo ""; echo "=== Done. End: $(date) ==="
