#!/bin/bash
#SBATCH --job-name=extract_mlm
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/extract_mlm_%j.out

echo "=== Extract Akkadian MLM embeddings for SEAL ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/seal__embed

python v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py \
    || { echo "FAILED"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
