#!/bin/bash
#SBATCH --job-name=quick_eda
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/linear_probing/logs/quick_eda_%j.out

echo "=== Quick EDA (Final Layer) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

python v_1/src/linear_probing/00b_quick_eda.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    || { echo "FAILED: quick EDA"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
