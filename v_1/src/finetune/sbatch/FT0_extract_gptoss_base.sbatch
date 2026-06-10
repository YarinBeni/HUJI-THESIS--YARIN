#!/bin/bash
#SBATCH --job-name=ft_FT0_gptoss_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/finetune/logs/FT0_extract_%j.out

# ===========================================================================
# FT0 — extract gpt-oss-120b BASE activations on the ORCC corpus.
# This IS meeting Task 2 ("try gpt-oss, the big one") and gives the fig4
# layerwise curve we need to sanity-check the depth-ablation cut points.
# tier0 + maximal, mean pooling, all 37 hidden states (36 blocks + embed).
# 4x H100: covers the bf16-dequantized case (~240GB); if MXFP4 kernels are
# available it simply uses less memory. Probing runs in FT0b (CPU).
# ===========================================================================
set -uo pipefail
echo "=== FT0 gpt-oss-120b base extraction ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs

ACTS=v_1/src/linear_probing/results/orcc__embed/activations

for CLEAN in tier0 maximal; do
    python -u v_1/src/linear_probing/03_extract_seal_activations.py \
        --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
        --text-col text_${CLEAN} \
        --model openai/gpt-oss-120b \
        --pooling mean \
        --batch-size 2 \
        --output-dir ${ACTS}/gpt_oss_120b_${CLEAN}_mean \
        || { echo "FAILED: extract ${CLEAN}"; exit 1; }
done

git add ${ACTS}/gpt_oss_120b_*_mean/metadata.json 2>/dev/null || true
git commit -m "finetune FT0: gpt-oss-120b base extraction metadata (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. Now submit FT0b_probe_gptoss_base.sbatch  $(date) ==="
