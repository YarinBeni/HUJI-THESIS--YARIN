#!/bin/bash
#SBATCH --job-name=maxfig_M1b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M1b_%j.out

# ===========================================================================
# M1b — RE-RENDER fig1(A,C,D)/fig2(A,B)/fig4(A) ONLY, from the probe summaries
# M1 already wrote. No recompute. Picks up the filter fix so tfidf (pool=na)
# and mlm (no pool token) now appear, plus tfidf-as-single-layer line in fig4.
# Seconds of work; the wait is just the queue.
# ===========================================================================
set -uo pipefail
echo "=== M1b re-render ===  Job ${SLURM_JOB_ID:-NA}  $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"

MF=v_1/src/geodesic/maximal_figs
python -u "$MF/make_maximal_figs.py" \
    --probes-dir "$MF/probes" --tag mc_balanced_maximal \
    --tables-out "$MF/tables" --fig-out "$MF/figures"

echo "=== Committing ==="
git add "$MF/figures"/*.png "$MF/tables"/*.csv "$MF/make_maximal_figs.py"
git commit -m "maximal_figs: re-render fig1/2/4 with tfidf+mlm included (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
