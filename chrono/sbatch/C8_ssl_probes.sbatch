#!/bin/bash
#SBATCH --job-name=C8_ssl_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=chrono/sbatch/logs/C8_ssl_probe_%A_%a.out
# ===========================================================================
# C8 — S1 representation probes (CPU) on the C7 embeddings: period / genre /
# provenance / SOURCE probes, within-source and held-out-source period probes,
# silhouette + permutation null, k-NN purity, UMAP. One task per (model, layer).
# Needs: afterok on the C7 array.
# ===========================================================================
set -uo pipefail
CELLS=("Thalesian/cuneiformBase-400m 6" "Thalesian/cuneiformBase-400m 12" "Thalesian/AKK_300m 4" "Thalesian/AKK_300m 8"
       "NousResearch/Llama-2-7b-hf 16" "NousResearch/Llama-2-7b-hf 24" "Qwen/Qwen3-8B 18" "Qwen/Qwen3-8B 27")
read -r MODEL LAYER <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "=== C8 probes ${MODEL} L${LAYER} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/ssl
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1
python -u chrono/scripts/probe_representations.py --model "${MODEL}" --layer "${LAYER}" --site mean \
    --out-dir chrono/reports/ssl || { echo "FAILED probes ${MODEL} L${LAYER}"; exit 1; }
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C8: S1 probes ${MODEL} L${LAYER} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
