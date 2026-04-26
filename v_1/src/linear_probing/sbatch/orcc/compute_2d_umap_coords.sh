#!/bin/bash
#SBATCH --job-name=orcc_coords
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_coords_%j.out

echo "=== ORCC 2D Coords — mean + last, t-SNE / PCA / UMAP ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

pip install umap-learn --quiet

# Mean-pooled coords
python -u v_1/src/linear_probing/04_compute_2d_coords.py \
    --pooling mean --include-umap \
    --input-base v_1/src/linear_probing/results/orcc_round1/activations \
    --input-dirs qwen_tier0_mean qwen_maximal_mean random_tier0_mean random_maximal_mean \
    --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
    --output-path v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_mean.json \
    || { echo "FAILED: orcc mean coords"; exit 1; }

# Last-token coords
python -u v_1/src/linear_probing/04_compute_2d_coords.py \
    --pooling last --include-umap \
    --input-base v_1/src/linear_probing/results/orcc_round1/activations \
    --input-dirs qwen_tier0_last qwen_maximal_last random_tier0_last random_maximal_last \
    --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
    --output-path v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_last.json \
    || { echo "FAILED: orcc last coords"; exit 1; }

echo "=== Pushing results to GitHub ==="
git add \
    v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_mean.json \
    v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_last.json
git commit -m "Add ORCC coord JSONs: mean+last t-SNE/PCA/UMAP (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally at results/orcc_round1/"

echo "=== Done ==="
echo "End: $(date)"
