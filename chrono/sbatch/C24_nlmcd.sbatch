#!/bin/bash
#SBATCH --job-name=C24_nlmcd
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=chrono/sbatch/logs/C24_nlmcd_%j.out
# ===========================================================================
# C24 / P6.4 — NLMCD-style concept manifolds (arXiv:2610.01821 adapted):
# UMAP + HDBSCAN over 4 stored representations, clusters labelled by
# metadata, period substructure inside source-dominated clusters, adjusted-
# Rand alignment across representations. CPU only; reads the emb store.
# ===========================================================================
set -uo pipefail
echo "=== C24 NLMCD === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1
python -u chrono/scripts/nlmcd_concepts.py || { echo "FAILED nlmcd"; exit 1; }
commit_push_sandbox "C24: NLMCD concept manifolds (job ${SLURM_JOB_ID})" chrono/reports/ssl
echo "=== Done. $(date) ==="
