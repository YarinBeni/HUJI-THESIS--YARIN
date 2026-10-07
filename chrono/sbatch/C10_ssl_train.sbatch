#!/bin/bash
#SBATCH --job-name=C10_ssl
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=chrono/sbatch/logs/C10_ssl_%A_%a.out
# ===========================================================================
# C10 — S2 SSL grid: {barlow, byol, jepa, infonce} x {cunei400m, akk300m,
# llama2_7b, qwen3_8b}, adapter on frozen features (C9), source-balanced, no
# labels; then the S1 probe battery on the learned embeddings so every run is
# scored the same way as the frozen encoders. Needs: afterok on C9.
# ===========================================================================
set -uo pipefail
OBJS=(barlow byol jepa infonce); ENCS=(cunei400m akk300m llama2_7b qwen3_8b)
OBJ=${OBJS[$((SLURM_ARRAY_TASK_ID / 4))]}; ENC=${ENCS[$((SLURM_ARRAY_TASK_ID % 4))]}
RUN="ssl_${OBJ}_${ENC}"
echo "=== C10 ${RUN} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl/heads
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1
python -u chrono/scripts/train_ssl.py --config "chrono/configs/ssl/${RUN}.yaml" --seed 0 \
    || { echo "FAILED train_ssl ${RUN}"; exit 1; }
cp "chrono/artifacts_ssl/ssl_runs/${RUN}-s0.pt" chrono/reports/ssl/heads/ 2>/dev/null || true
python -u chrono/scripts/probe_representations.py --model "ssl::${RUN}-s0" --layer 0 --site h --pca 0 \
    --out-dir chrono/reports/ssl || echo "WARN probes failed ${RUN}"
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C10: ${RUN} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
