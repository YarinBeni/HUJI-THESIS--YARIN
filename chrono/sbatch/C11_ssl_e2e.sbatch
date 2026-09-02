#!/bin/bash
#SBATCH --job-name=C11_e2e
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:30:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C11_e2e_%A_%a.out
# ===========================================================================
# C11 — S2 from-scratch family: SignEncoder {S 8M, M 21M, L 47M, XL 100M} x
# {barlow, jepa}, views generated on the fly, source-balanced, 5 wall-clock
# hours each (resumable from ckpt.pt). Quick period probe every 2k steps in
# the log; final embeddings -> store 'ssl_e2e::<run>' for the C12 battery.
# ===========================================================================
set -uo pipefail
SIZES=(S M L XL); OBJS=(barlow jepa)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID % 4))]}; OBJ=${OBJS[$((SLURM_ARRAY_TASK_ID / 4))]}
BATCH=256; [ "$SIZE" = "XL" ] && BATCH=192
echo "=== C11 e2e ${OBJ} ${SIZE} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl/e2e
ART=chrono/artifacts_ssl
( flock -w 1200 9
  [ -f "${ART}/corpus_all.parquet" ] || python -u chrono/scripts/make_ssl_corpus.py --out-dir "${ART}" || exit 1
) 9>"${GIT_LOCK}.ssl" || { echo "FAILED corpus build"; exit 1; }
python -u chrono/scripts/train_ssl_e2e.py --size "${SIZE}" --objective "${OBJ}" --hours 5 --batch "${BATCH}" \
    --corpus "${ART}/corpus_all.parquet" --store-root "${ART}/emb_store" --out-dir "${ART}/e2e" \
    || { echo "FAILED e2e ${OBJ} ${SIZE}"; exit 1; }
cp "${ART}/e2e/e2e_${OBJ}_${SIZE}-s0/final.pt" "chrono/reports/ssl/e2e/e2e_${OBJ}_${SIZE}-s0.pt" 2>/dev/null || true
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C11: e2e ${OBJ} ${SIZE} (job ${SLURM_JOB_ID})" chrono/reports/results.parquet
echo "=== Done. $(date) ==="
