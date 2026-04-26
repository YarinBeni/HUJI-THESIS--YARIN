#!/bin/bash
#SBATCH --job-name=seal_mlm_umap
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/seal_mlm_umap_%j.out

echo "=== SEAL MLM Embeddings — recompute with UMAP ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

pip install umap-learn --quiet

python -u v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py \
    --include-umap \
    --output-path v_1/src/linear_probing/results/seal_round4/seal_mlm_coords.json \
    || { echo "FAILED: seal mlm umap"; exit 1; }

echo "=== Pushing results to GitHub ==="
git add v_1/src/linear_probing/results/seal_round4/seal_mlm_coords.json
git commit -m "Add SEAL MLM UMAP coords (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo "=== Done ==="
echo "End: $(date)"
