#!/bin/bash
#SBATCH --job-name=seal_coords_last
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=v_1/src/linear_probing/logs/seal_coords_last_%j.out

echo "=== SEAL 2D Coords — last-token + UMAP ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

pip install umap-learn --quiet

python -u v_1/src/linear_probing/04_compute_2d_coords.py \
    --pooling last \
    --include-umap \
    --input-base v_1/src/linear_probing/results/seal__embed/activations \
    --input-dirs qwen_tier0_last qwen_maximal_last random_tier0_last random_maximal_last \
    --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
    --output-path v_1/src/linear_probing/results/seal__embed/seal_qwen_coords_last.json \
    || { echo "FAILED: seal coords last"; exit 1; }

echo "=== Pushing results to GitHub ==="
git add v_1/src/linear_probing/results/seal__embed/seal_qwen_coords_last.json
git commit -m "Add SEAL last-token coord JSON: t-SNE/PCA/UMAP (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally at results/seal__embed/"

echo "=== Done ==="
echo "End: $(date)"
