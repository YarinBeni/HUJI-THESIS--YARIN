#!/bin/bash
#SBATCH --job-name=C21_e2e_wd
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:30:00
#SBATCH --output=chrono/sbatch/logs/C21_e2e_wd_%j.out
# C21 — from-scratch barlow S (its family's best) with the dated text included.
set -uo pipefail
echo "=== C21 e2e barlow S wdated === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
python -u chrono/scripts/train_ssl_e2e.py --size S --objective barlow --hours 5 --batch 256 \
    --include-dated || { echo "FAILED e2e wdated"; exit 1; }
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C21: e2e_barlow_S_wdated (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
