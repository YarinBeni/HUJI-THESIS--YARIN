#!/bin/bash
#SBATCH --job-name=tok_check
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/tok_check_%j.out

echo "=== Tokenization Check ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

python v_1/src/linear_probing/00_tokenization_check.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    || { echo "FAILED: tokenization check"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
