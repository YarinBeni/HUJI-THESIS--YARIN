#!/bin/bash
#SBATCH --job-name=C20b_seed
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=chrono/sbatch/logs/C20b_seed_%j.out
# C20b — seed replication of the winning wdated arms (WD_OBJ, WD_SEED env).
set -uo pipefail
echo "=== C20b ${WD_OBJ} wdated anti=${WD_ANTI:-none} seed ${WD_SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
EXTRA=(); [ -n "${WD_ANTI:-}" ] && EXTRA=(--anti "${WD_ANTI}")
python -u chrono/scripts/train_ssl.py --config "chrono/configs/ssl/ssl_${WD_OBJ}_cunei400m.yaml" \
    --seed "${WD_SEED}" --include-dated "${EXTRA[@]}" || { echo "FAILED train"; exit 1; }
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C20b: ssl_${WD_OBJ}_cunei400m_wdated-s${WD_SEED} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
