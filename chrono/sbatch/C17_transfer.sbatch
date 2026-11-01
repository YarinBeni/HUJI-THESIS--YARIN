#!/bin/bash
#SBATCH --job-name=C17_transfer
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=chrono/sbatch/logs/C17_transfer_%j.out
# ===========================================================================
# C17 — the transfer read-out the earlier probes could not give: fit on the
# undated corpora, evaluate on the dated royal inscriptions. One job covers
# every cell in the store, so it costs one slot instead of 32.
# ===========================================================================
set -uo pipefail
echo "=== C17 transfer === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
python -u chrono/scripts/probe_transfer_dated.py || { echo "FAILED transfer probe"; exit 1; }
commit_push_sandbox "C17: transfer to the dated inscriptions (job ${SLURM_JOB_ID})" chrono/reports/ssl
echo "=== Done. $(date) ==="
