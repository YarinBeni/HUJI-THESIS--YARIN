#!/bin/bash
#SBATCH --job-name=orcc_mlm
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_mlm_%j.out

echo "=== ORCC MLM Embeddings — tsne/pca/umap ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed

pip install umap-learn --quiet

python -u v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --include-umap \
    --output-path v_1/src/linear_probing/results/orcc__embed/orcc_mlm_coords.json \
    || { echo "FAILED: orcc mlm"; exit 1; }

echo "=== Pushing results to GitHub ==="
git add v_1/src/linear_probing/results/orcc__embed/orcc_mlm_coords.json
git commit -m "Add ORCC MLM coords: tsne/pca/umap (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo "=== Done ==="
echo "End: $(date)"
