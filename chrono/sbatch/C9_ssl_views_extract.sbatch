#!/bin/bash
#SBATCH --job-name=C9_ssl_views
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=chrono/sbatch/logs/C9_ssl_views_%A_%a.out
# ===========================================================================
# C9 — S2 features: embed all 351k SSL views (make_ssl_views.py) with the four
# frozen encoders at the S2 layer, sharded over array tasks so each runs ~1h:
# cunei400m & AKK_300m 2 parts each, Llama-2-7B & Qwen3-8B 8 parts each.
# Store ids 'ssl::<view_id>'. Resume-safe (the store skips cached shards).
# ===========================================================================
set -uo pipefail
# task -> (model, layer, part k/n)
if   [ "$SLURM_ARRAY_TASK_ID" -lt 2 ];  then MODEL=thalesian_cunei400m; LAYER=12; K=$SLURM_ARRAY_TASK_ID; N=2
elif [ "$SLURM_ARRAY_TASK_ID" -lt 4 ];  then MODEL=thalesian_akk300m;  LAYER=8;  K=$((SLURM_ARRAY_TASK_ID-2)); N=2
elif [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then MODEL=llama2_7b;          LAYER=16; K=$((SLURM_ARRAY_TASK_ID-4)); N=8
else                                          MODEL=qwen3_8b;           LAYER=18; K=$((SLURM_ARRAY_TASK_ID-12)); N=8; fi
echo "=== C9 views ${MODEL} L${LAYER} part ${K}/${N} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
ART=chrono/artifacts_ssl
( flock -w 1200 9
  [ -f "${ART}/corpus_all.parquet" ] || python -u chrono/scripts/make_ssl_corpus.py --out-dir "${ART}" || exit 1
  [ -f "${ART}/views_ssl.parquet" ] || python -u chrono/scripts/make_ssl_views.py --corpus "${ART}/corpus_all.parquet" --out "${ART}/views_ssl.parquet" || exit 1
) 9>"${GIT_LOCK}.ssl" || { echo "FAILED corpus/views build"; exit 1; }
python -u chrono/scripts/extract_embeddings.py --model "${MODEL}" --layers "${LAYER}" --sites mean \
    --table "${ART}/views_ssl.parquet" --id-col view_id --text-col text --id-prefix "ssl::" --part "${K}/${N}" \
    --store-root "${ART}/emb_store" --batch-size 16 --max-tokens 512 \
    || { echo "FAILED extract ${MODEL} part ${K}"; exit 1; }
echo "=== Done. $(date) ==="
