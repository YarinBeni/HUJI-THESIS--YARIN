#!/bin/bash
#SBATCH --job-name=C1v2_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=chrono/sbatch/logs/C1v2_extract_%A_%a.out
# ===========================================================================
# C1v2 — E-MIN v2 features (review 2026-09-02): tier0 Akkadian corpus in its
# OWN artifacts root (chrono/artifacts_tier0: corpus/splits/views/store all
# rebuilt there, because tier0 has 1,193 docs vs maximal's 1,187 and the
# folds must match the corpus), embedded by three encoders, one per array
# task: the two LLMs the M.Sc. found best on documents (Llama-2-7B on the
# Akkadian, Qwen3-8B on the gloss) at three mid/late layers, and the
# stronger Akkadian encoder cuneiformBase-400m at every layer.
# ===========================================================================
set -uo pipefail
MODELS=(llama2_7b qwen3_8b thalesian_cunei400m)
LAYERS=("8,16,24,32" "9,18,27,36" "all")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}; LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
echo "=== C1v2 extract ${MODEL} layers ${LAYER} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs

ART=chrono/artifacts_tier0
mkdir -p "${ART}"
# the three deterministic artifacts are built once; array tasks race on the
# same files, so build under the git lock and only if absent
( flock -w 900 9
  if [ ! -f "${ART}/corpus_chrono.parquet" ]; then
      python -u chrono/scripts/make_corpus.py --akk-tier tier0 --out-dir "${ART}" \
          || { echo "FAILED make_corpus tier0"; exit 1; }
  fi
  if [ ! -f "${ART}/splits/gkf_ruler.json" ]; then
      python -u chrono/scripts/make_splits.py --corpus "${ART}/corpus_chrono.parquet" \
          --out-dir "${ART}/splits" || { echo "FAILED make_splits"; exit 1; }
  fi
  if [ ! -f "${ART}/views.parquet" ]; then
      python -u chrono/scripts/make_views.py --corpus "${ART}/corpus_chrono.parquet" \
          --out "${ART}/views.parquet" --audit-out "${ART}/views_audit.parquet" \
          --seeds 0 1 || { echo "FAILED make_views"; exit 1; }
  fi
) 9>"${GIT_LOCK}.tier0" || exit 1

python -u chrono/scripts/extract_embeddings.py \
    --model "${MODEL}" --layers "${LAYER}" --sites mean last \
    --views "${ART}/views.parquet" --corpus "${ART}/corpus_chrono.parquet" \
    --store-root "${ART}/emb_store" --batch-size 16 --max-tokens 512 \
    || { echo "FAILED extract_embeddings ${MODEL}"; exit 1; }

mkdir -p chrono/reports/tier0
cp "${ART}/emb_store/extract_meta__${MODEL}.json" chrono/reports/tier0/
commit_push_sandbox "C1v2: tier0 features ${MODEL} (job ${SLURM_JOB_ID})" chrono/reports/tier0
echo "=== Done. $(date) ==="
