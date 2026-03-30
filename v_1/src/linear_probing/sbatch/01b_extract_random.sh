#!/bin/bash
#SBATCH --job-name=extract_random
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/extract_random_%j.out

echo "=== Extract Random-Weights Baseline Activations ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

# --- Mean pooling ---
echo "=== Extracting random activations (tier0, mean pooling) ==="
python -u v_1/src/linear_probing/01b_extract_random_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning tier0 \
    --pooling mean \
    --batch-size 8 \
    || { echo "FAILED: tier0 mean extraction"; exit 1; }

echo "=== Extracting random activations (maximal, mean pooling) ==="
python -u v_1/src/linear_probing/01b_extract_random_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning maximal \
    --pooling mean \
    --batch-size 8 \
    || { echo "FAILED: maximal mean extraction"; exit 1; }

# --- Last-token pooling ---
echo "=== Extracting random activations (tier0, last_token pooling) ==="
python -u v_1/src/linear_probing/01b_extract_random_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning tier0 \
    --pooling last_token \
    --batch-size 8 \
    || { echo "FAILED: tier0 last_token extraction"; exit 1; }

echo "=== Extracting random activations (maximal, last_token pooling) ==="
python -u v_1/src/linear_probing/01b_extract_random_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning maximal \
    --pooling last_token \
    --batch-size 8 \
    || { echo "FAILED: maximal last_token extraction"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
