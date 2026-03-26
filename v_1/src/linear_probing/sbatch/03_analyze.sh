#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/analyze_%j.out

echo "=== Analyze Results ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

python v_1/src/linear_probing/03_analyze_results.py \
    --model qwen2.5-7b-instruct \
    || { echo "FAILED: analyze"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
