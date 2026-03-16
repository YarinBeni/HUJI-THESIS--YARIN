#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=test_qwen_%j.out

echo "=== Starting model load test ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate environment
source ~/miniconda3/bin/activate thesis

# Run the test
python ~/projects/lititure-review/v_1/src/cluster/test_model_load.py

echo "=== Done ==="
