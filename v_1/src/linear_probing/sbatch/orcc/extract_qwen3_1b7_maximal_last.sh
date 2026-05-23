#!/bin/bash
#SBATCH --job-name=r3_e1_1b7_mx_lst
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_1b7_mx_lst_%j.out

echo "=== Phase E1: Qwen3-1.7B maximal last ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations/qwen3_1b7_maximal_last

python -u v_1/src/linear_probing/03_extract_seal_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_maximal \
    --model Qwen/Qwen3-1.7B \
    --pooling last \
    --batch-size 16 \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/qwen3_1b7_maximal_last \
    || { echo "FAILED: qwen3_1b7_maximal_last"; exit 1; }

git add v_1/src/linear_probing/results/orcc__embed/activations/qwen3_1b7_maximal_last/ && git commit -m "Phase E1: qwen3_1b7_maximal_last activations (job $SLURM_JOB_ID)" || true && git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="
echo "End: $(date)"
