#!/bin/bash
#SBATCH --job-name=C19_anti
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C19_anti_%A_%a.out
# ===========================================================================
# C19 — S5b anti-shortcut arms (advisor, 2026-09-04). The 32-run sweep showed
# every SSL family learns SOURCE (probe .92-.98) and never beats the frozen
# encoder on dating. Forbid the shortcut during training, on the best combo
# (cunei400m adapters): {barlow, jepa} x {leace-in-the-loop, adversarial}.
# Success check no.1 is the SOURCE probe dropping from ~.97; the dating
# read-out is C18. Probes run in-job; a C18 refresh is chained by inbox 043.
# ===========================================================================
set -uo pipefail
OBJS=(barlow jepa); ANTIS=(leace adv both leopard)
OBJ=${OBJS[$((SLURM_ARRAY_TASK_ID / 4))]}; ANTI=${ANTIS[$((SLURM_ARRAY_TASK_ID % 4))]}
echo "=== C19 ${OBJ} + ${ANTI} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
# smoke first: 60 steps must run clean before the real thing burns hours
python -u chrono/scripts/train_ssl.py --config "chrono/configs/ssl/ssl_${OBJ}_cunei400m.yaml" \
    --seed 99 --steps 60 --anti "${ANTI}" || { echo "FAILED smoke ${OBJ} ${ANTI}"; exit 1; }
python -u chrono/scripts/train_ssl.py --config "chrono/configs/ssl/ssl_${OBJ}_cunei400m.yaml" \
    --seed 0 --anti "${ANTI}" || { echo "FAILED train ${OBJ} ${ANTI}"; exit 1; }
RUN="ssl_${OBJ}_cunei400m_${ANTI}-s0"
python -u chrono/scripts/probe_representations.py --model "ssl::${RUN}" --layer 0 --site h --pca 0 \
    --out-dir chrono/reports/ssl || echo "WARN probes failed ${RUN}"
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C19: ${RUN} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
