#!/bin/bash
#SBATCH --job-name=r3_e1_32b_t0_lt
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_32b_t0_lt_%j.out

echo "=== Phase E1: Qwen3-32B tier0 last ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations/qwen3_32b_tier0_last

python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --model Qwen/Qwen3-32B \
    --pooling last \
    --batch-size 2 \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen3_32b_tier0_last \
    || { echo "FAILED: qwen3_32b_tier0_last"; exit 1; }

git add v_1/src/linear_probing/results/orcc__embed/activations/qwen3_32b_tier0_last/ && git commit -m "Phase E1: qwen3_32b_tier0_last activations (job $SLURM_JOB_ID)" || true && git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="
echo "End: $(date)"
