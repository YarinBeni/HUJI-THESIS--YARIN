#!/bin/bash
#SBATCH --job-name=r3_e1_27b_t0_lt
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_27b_t0_lt_%j.out

echo "=== Phase E1: Qwen3.5-27B tier0 last ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_tier0_last

python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --model Qwen/Qwen3.5-27B-Instruct \
    --pooling last \
    --batch-size 4 \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_tier0_last \
    || { echo "FAILED: qwen35_27b_tier0_last"; exit 1; }

git add v_1/src/linear_probing/results/orcc__embed/activations/qwen35_27b_tier0_last/ && git commit -m "Phase E1: qwen35_27b_tier0_last activations (job $SLURM_JOB_ID)" || true && git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="
echo "End: $(date)"
