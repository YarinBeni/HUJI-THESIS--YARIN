#!/bin/bash
#SBATCH --job-name=extract_acts
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/extract_%j.out

echo "=== Extract Activations ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

echo "=== Extracting activations (tier0) ==="
python v_1/src/linear_probing/01_extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning tier0 \
    --batch-size 8 \
    || { echo "FAILED: tier0 extraction"; exit 1; }

echo "=== Extracting activations (maximal cleaning) ==="
python v_1/src/linear_probing/01_extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cleaning maximal \
    --batch-size 8 \
    || { echo "FAILED: maximal extraction"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
