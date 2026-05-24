#!/bin/bash
#SBATCH --job-name=r3_e1_27b_mx_mn
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_27b_mx_mn_%j.out

echo "=== Phase E1: Qwen3.5-27B maximal mean ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_maximal_mean

python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_maximal \
    --model Qwen/Qwen3.5-27B-Instruct \
    --pooling mean \
    --batch-size 4 \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_maximal_mean \
    || { echo "FAILED: qwen35_27b_maximal_mean"; exit 1; }

git add v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_maximal_mean/ && git commit -m "Phase E1: qwen35_27b_maximal_mean activations (job $SLURM_JOB_ID)" || true && git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="
echo "End: $(date)"
