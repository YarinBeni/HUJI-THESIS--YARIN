#!/bin/bash
#SBATCH --job-name=C26_deep_ext
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=chrono/sbatch/logs/C26_deep_ext_%A_%a.out
# ===========================================================================
# C26 — deep-layer extraction over the tier0 views so the atlas can draw the
# layer x layer panels for the LLMs too (the store holds only a 4-layer
# ladder for them; cunei400m already has all layers). Site mean only.
# ===========================================================================
set -uo pipefail
MODELS=(llama2_7b qwen3_8b olmo2_7b)
LAYERS=("2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32" \
        "3,6,9,12,15,18,21,24,27,30,33,36" \
        "3,6,9,12,15,18,21,24,27,30,32")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}; LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
echo "=== C26 deep extract ${MODEL} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
ART=chrono/artifacts_tier0
python -u chrono/scripts/extract_embeddings.py \
    --model "${MODEL}" --layers "${LAYER}" --sites mean \
    --store-root "${ART}/emb_store" --batch-size 16 --max-tokens 512 \
    || { echo "FAILED extract ${MODEL}"; exit 1; }
commit_push_sandbox "C26: deep layers ${MODEL} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
