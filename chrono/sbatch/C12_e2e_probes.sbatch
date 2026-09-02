#!/bin/bash
#SBATCH --job-name=C12_e2e_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C12_e2e_probe_%A_%a.out
# C12 — S1 probe battery on the from-scratch SSL encoders (C11). afterok C11.
set -uo pipefail
SIZES=(S M L XL); OBJS=(barlow jepa)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID % 4))]}; OBJ=${OBJS[$((SLURM_ARRAY_TASK_ID / 4))]}
RUN="e2e_${OBJ}_${SIZE}-s0"
echo "=== C12 probes ${RUN} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1
python -u chrono/scripts/probe_representations.py --model "ssl_e2e::${RUN}" --layer 0 --site h --pca 0 \
    --store-root chrono/artifacts_ssl/emb_store --out-dir chrono/reports/ssl || { echo "FAILED probes ${RUN}"; exit 1; }
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C12: e2e probes ${RUN} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
