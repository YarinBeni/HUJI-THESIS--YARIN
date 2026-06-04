#!/bin/bash
#SBATCH --job-name=fig1_bestk
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=v_1/src/geodesic/fig1_followups/pls_ksweep/logs/bestk_%j.out

# ===========================================================================
# Re-aggregate the EXISTING k-sweep draws (no re-run) to reproduce Fig-1A's
# best-k-per-draw PLS and compare it to fixed-k + Ridge. The raw 600+ per-draw
# JSONs live only on the cluster, hence this runs here. Commits + pushes the
# small figure + CSV. Seconds of work.
# ===========================================================================

set -uo pipefail
echo "=== Fig-1: best-k-per-draw vs fixed-k vs Ridge ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

FU=v_1/src/geodesic/fig1_followups
mkdir -p "$FU/pls_ksweep/logs"

python -u "$FU/pls_ksweep/best_k_compare.py" \
    --probes-dir   "$FU/pls_ksweep/probes" \
    --tradeoff-csv "$FU/pls_ksweep/pls_components_tradeoff.csv" \
    --fixed-k 3 || { echo "FAILED"; exit 1; }

echo ""; echo "=== Committing ==="
git add "$FU/pls_ksweep/best_k_vs_fixed_k.png" "$FU/pls_ksweep/best_k_vs_fixed_k.csv"
git commit -m "Fig-1: best-k-per-draw vs fixed-k vs Ridge (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN: pull --rebase failed"
git push origin main || echo "WARN: push failed — outputs are on the cluster filesystem."

echo ""; echo "=== Done. End: $(date) ==="
