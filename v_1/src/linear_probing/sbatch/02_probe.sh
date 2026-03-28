#!/bin/bash
#SBATCH --job-name=lin_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/probe_%j.out

echo "=== Linear Probe ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

echo "=== Probing (mean pooling) ==="
python -u v_1/src/linear_probing/02_linear_probe.py \
    --model qwen2.5-7b-instruct \
    --pooling mean \
    --n-permutations 200 \
    || { echo "FAILED: linear probe (mean)"; exit 1; }

echo "=== Probing (last_token pooling) ==="
python -u v_1/src/linear_probing/02_linear_probe.py \
    --model qwen2.5-7b-instruct \
    --pooling last_token \
    --n-permutations 200 \
    || { echo "FAILED: linear probe (last_token)"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
