#!/bin/bash
#SBATCH --job-name=C16_probe_hyb
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C16_probe_hyb_%A_%a.out
# ===========================================================================
# C16 — S1 probe battery for the hybrid family (C15). afterok C15.
# (was: C14 — re-run the S1 probe battery for EVERY embedding cell now that the
# output path is filename-safe (the C10/C12 probes computed everything and
# died writing 'umap_ssl::...parquet'). Cells: 4 frozen encoders x 2 layers,
# 16 SSL adapters, 8 from-scratch encoders. Missing cells exit 0 quietly.
# ===========================================================================
set -uo pipefail
CELLS=()
for FRZ in thalesian_cunei400m llama2_7b; do for SZ in S M; do for OBJ in barlow jepa; do
  CELLS+=("ssl_hyb::hyb_${OBJ}_${SZ}_${FRZ}-s0 0 h 0"); done; done; done
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
