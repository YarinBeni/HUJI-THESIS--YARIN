#!/bin/bash
#SBATCH --job-name=C7_ssl_ext
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=chrono/sbatch/logs/C7_ssl_ext_%A_%a.out
# ===========================================================================
# C7 — S1 of PLAN_SCALE_SSL: embed EVERY text of the SSL corpus (35k texts,
# all sources) with the four frozen encoders, into chrono/artifacts_ssl/
# emb_store (ids 'ssl::<fragment_id>'). One array task per encoder. The
# corpus itself is rebuilt on the cluster (deterministic; ~2 min).
# ===========================================================================
set -uo pipefail
MODELS=(thalesian_cunei400m thalesian_akk300m llama2_7b qwen3_8b)
LAYERS=("6,12" "4,8" "16,24" "18,27")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}; LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
echo "=== C7 ssl extract ${MODEL} L${LAYER} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
ART=chrono/artifacts_ssl
( flock -w 900 9
  if [ ! -f "${ART}/corpus_all.parquet" ]; then
      python -u chrono/scripts/make_ssl_corpus.py --out-dir "${ART}" || { echo "FAILED make_ssl_corpus"; exit 1; }
      cp "${ART}/CENSUS.md" chrono/reports/ssl/CENSUS.md
  fi
) 9>"${GIT_LOCK}.ssl" || exit 1
python -u chrono/scripts/extract_embeddings.py --model "${MODEL}" --layers "${LAYER}" --sites mean \
    --table "${ART}/corpus_all.parquet" --id-col fragment_id --text-col text --id-prefix "ssl::" \
    --store-root "${ART}/emb_store" --batch-size 16 --max-tokens 512 \
    || { echo "FAILED extract ${MODEL}"; exit 1; }
cp "${ART}/emb_store/extract_meta__${MODEL}.json" chrono/reports/ssl/
commit_push_sandbox "C7: ssl features ${MODEL} (job ${SLURM_JOB_ID})" chrono/reports/ssl
echo "=== Done. $(date) ==="
