#!/bin/bash
#SBATCH --job-name=C14_reprobe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=chrono/sbatch/logs/C14_reprobe_%A_%a.out
# ===========================================================================
# C14 — re-run the S1 probe battery for EVERY embedding cell now that the
# output path is filename-safe (the C10/C12 probes computed everything and
# died writing 'umap_ssl::...parquet'). Cells: 4 frozen encoders x 2 layers,
# 16 SSL adapters, 8 from-scratch encoders. Missing cells exit 0 quietly.
# ===========================================================================
set -uo pipefail
CELLS=(
  "Thalesian/cuneiformBase-400m 6 mean 256" "Thalesian/cuneiformBase-400m 12 mean 256"
  "Thalesian/AKK_300m 4 mean 256" "Thalesian/AKK_300m 8 mean 256"
  "NousResearch/Llama-2-7b-hf 16 mean 256" "NousResearch/Llama-2-7b-hf 24 mean 256"
  "Qwen/Qwen3-8B 18 mean 256" "Qwen/Qwen3-8B 27 mean 256"
)
for OBJ in barlow byol jepa infonce; do for ENC in cunei400m akk300m llama2_7b qwen3_8b; do
  CELLS+=("ssl::ssl_${OBJ}_${ENC}-s0 0 h 0"); done; done
for OBJ in barlow jepa; do for SZ in S M L XL; do
  CELLS+=("ssl_e2e::e2e_${OBJ}_${SZ}-s0 0 h 0"); done; done
[ "$SLURM_ARRAY_TASK_ID" -ge "${#CELLS[@]}" ] && { echo "no cell"; exit 0; }
read -r MODEL LAYER SITE PCA <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "=== C14 ${MODEL} L${LAYER} ${SITE} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1
python -u chrono/scripts/probe_representations.py --model "${MODEL}" --layer "${LAYER}" --site "${SITE}" --pca "${PCA}" \
    --store-root chrono/artifacts_ssl/emb_store --out-dir chrono/reports/ssl \
    || { echo "SKIP/FAIL ${MODEL} (cell may not be extracted yet)"; exit 0; }
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C14: probes ${MODEL} L${LAYER} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
