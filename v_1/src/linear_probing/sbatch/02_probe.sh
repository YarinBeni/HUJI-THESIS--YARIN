#!/bin/bash
#SBATCH --job-name=lin_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

python v_1/src/linear_probing/02_linear_probe.py \
    --model llama-3.1-8b-instruct \
    --n-permutations 1000 \
    || { echo "FAILED: linear probe"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
