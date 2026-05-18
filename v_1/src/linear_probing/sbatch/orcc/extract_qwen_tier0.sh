#!/bin/bash
#SBATCH --job-name=orcc_qwen_tier0
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_qwen_tier0_%j.out

echo "=== ORCC Qwen Activations — tier0 (mean + last) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations

# Mean pooling pass
python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --pooling mean \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen_tier0_mean \
    || { echo "FAILED: orcc qwen tier0 mean"; exit 1; }

# Last-token pooling pass
python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --pooling last \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen_tier0_last \
    || { echo "FAILED: orcc qwen tier0 last"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
