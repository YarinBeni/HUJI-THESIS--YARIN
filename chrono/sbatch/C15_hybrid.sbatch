#!/bin/bash
#SBATCH --job-name=C15_hyb
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:30:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C15_hyb_%A_%a.out
# ===========================================================================
# C15 — hybrid family (PI, 2026-09-03): the frozen encoder's TOKEN states feed
# a fresh S/M Transformer trained with SSL on the fly. {cunei400m L12, Llama-2-7B
# L16} x {S, M} x {barlow, jepa}, 5 wall-clock hours each, resumable.
# Embeddings -> store 'ssl_hyb::<run>'; probes by C16.
# ===========================================================================
set -uo pipefail
FRZ=(thalesian_cunei400m llama2_7b); FL=(12 16); SIZES=(S M); OBJS=(barlow jepa)
F=$(( SLURM_ARRAY_TASK_ID / 4 )); R=$(( SLURM_ARRAY_TASK_ID % 4 )); SZ=${SIZES[$((R / 2))]}; OBJ=${OBJS[$((R % 2))]}
BATCH=256; [ "${FRZ[$F]}" = "llama2_7b" ] && BATCH=96
echo "=== C15 hyb ${OBJ} ${SZ} on ${FRZ[$F]} L${FL[$F]} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
ART=chrono/artifacts_ssl
( flock -w 1200 9; [ -f "${ART}/corpus_all.parquet" ] || python -u chrono/scripts/make_ssl_corpus.py --out-dir "${ART}" || exit 1 ) 9>"${GIT_LOCK}.ssl" || exit 1
python -u chrono/scripts/train_ssl_e2e.py --size "${SZ}" --objective "${OBJ}" --hours 5 --batch "${BATCH}" \
    --frozen "${FRZ[$F]}" --frozen-layer "${FL[$F]}" \
    --corpus "${ART}/corpus_all.parquet" --store-root "${ART}/emb_store" --out-dir "${ART}/hyb" \
    || { echo "FAILED hyb ${OBJ} ${SZ} ${FRZ[$F]}"; exit 1; }
RUN="hyb_${OBJ}_${SZ}_${FRZ[$F]}-s0"
# final.pt stays under ${ART} (ignored by git); copying it into chrono/reports
# put 100-400 MB blobs into every later `commit_push_sandbox chrono/reports`.
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C15: ${RUN} (job ${SLURM_JOB_ID})" chrono/reports/results.parquet
echo "=== Done. $(date) ==="
