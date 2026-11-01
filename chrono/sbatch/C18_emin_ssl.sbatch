#!/bin/bash
#SBATCH --job-name=C18_emin_ssl
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=chrono/sbatch/logs/C18_emin_ssl_%j.out
# ===========================================================================
# C18 — the read-out Yarin asked for: every representation in the store,
# scored on the 1,193 dated royal inscriptions with the THESIS protocol
# (ruler-grouped folds, mc_balanced draws, SLA §7 pooled rho), so the rows
# sit next to EMIN2_RESULT.md instead of next to a period-classification
# number. One job for all cells.
# ===========================================================================
set -uo pipefail
echo "=== C18 emin-ssl === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports
python -u chrono/scripts/emin_ssl_readout.py || { echo "FAILED emin_ssl_readout"; exit 1; }
commit_push_sandbox "C18: dated inscriptions on the thesis protocol (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
