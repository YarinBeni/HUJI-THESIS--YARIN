#!/bin/bash
#SBATCH --job-name=seal_random_tier0
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/linear_probing/logs/seal_random_tier0_%j.out

echo "=== SEAL Random-Weights Activations — tier0 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

python -u v_1/src/linear_probing/03b_extract_random_seal_activations.py \
    --text-col text_tier0 \
    || { echo "FAILED: random tier0 extraction"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
