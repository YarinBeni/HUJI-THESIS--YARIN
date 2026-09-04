#!/bin/bash
#SBATCH --job-name=C20_wdated
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=chrono/sbatch/logs/C20_wdated_%A_%a.out
# ===========================================================================
# C20 — S5a (advisor fallback, pre-approved 2026-09-04): every C19 arm failed
# the three-part gate, so bring the dated inscriptions' RAW TEXT (never labels)
# into SSL training — the thesis setting, where the encoder saw the documents
# and only kings are held out at read-out. Cells: plain {barlow, jepa, byol}
# + the erasure champion {jepa+leace}, all on cunei400m.
# ===========================================================================
set -uo pipefail
OBJS=(barlow jepa byol jepa); ANTIS=("" "" "" leace)
OBJ=${OBJS[$SLURM_ARRAY_TASK_ID]}; ANTI=${ANTIS[$SLURM_ARRAY_TASK_ID]}
echo "=== C20 ${OBJ} wdated anti='${ANTI}' === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
EXTRA=(); [ -n "$ANTI" ] && EXTRA=(--anti "$ANTI")
python -u chrono/scripts/train_ssl.py --config "chrono/configs/ssl/ssl_${OBJ}_cunei400m.yaml" \
    --seed 0 --include-dated "${EXTRA[@]}" || { echo "FAILED train ${OBJ} ${ANTI}"; exit 1; }
SUF=""; [ -n "$ANTI" ] && SUF="_${ANTI}"
RUN="ssl_${OBJ}_cunei400m${SUF}_wdated-s0"
python -u chrono/scripts/probe_representations.py --model "ssl::${RUN}" --layer 0 --site h --pca 0 \
    --out-dir chrono/reports/ssl || echo "WARN probes failed ${RUN}"
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C20: ${RUN} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
