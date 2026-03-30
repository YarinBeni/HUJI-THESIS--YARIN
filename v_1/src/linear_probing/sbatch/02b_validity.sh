#!/bin/bash
#SBATCH --job-name=validity_tests
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/linear_probing/logs/validity_%j.out

echo "=== Validity Tests ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

mkdir -p v_1/src/linear_probing/logs

# --- Step 1: Probe random-weights baseline (reuse existing 02_linear_probe.py) ---
echo "=== Probing random baseline (mean pooling) ==="
python -u v_1/src/linear_probing/02_linear_probe.py \
    --model qwen2.5-7b-instruct-random \
    --pooling mean \
    --n-permutations 1000 \
    || { echo "FAILED: random probe (mean)"; exit 1; }

echo "=== Probing random baseline (last_token pooling) ==="
python -u v_1/src/linear_probing/02_linear_probe.py \
    --model qwen2.5-7b-instruct-random \
    --pooling last_token \
    --n-permutations 1000 \
    || { echo "FAILED: random probe (last_token)"; exit 1; }

# --- Step 2: Run validity tests (learning curve, PCA, MLP, comparison) ---
echo "=== Validity tests (mean pooling) ==="
python -u v_1/src/linear_probing/02b_validity_tests.py \
    --model qwen2.5-7b-instruct \
    --pooling mean \
    || { echo "FAILED: validity tests (mean)"; exit 1; }

echo "=== Validity tests (last_token pooling) ==="
python -u v_1/src/linear_probing/02b_validity_tests.py \
    --model qwen2.5-7b-instruct \
    --pooling last_token \
    || { echo "FAILED: validity tests (last_token)"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
