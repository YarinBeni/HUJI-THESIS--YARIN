#!/bin/bash
#SBATCH --job-name=mlm_all_layers
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=v_1/src/linear_probing/logs/mlm_all_layers_%j.out

# CPU-only: MLM is 36.7M params and runs comfortably without GPU.
# Outputs: layer_NN.npz (*.npz is gitignored) — activations stay on cluster only.
# After this job completes, run 05_compute_pls_mlm.py to fit PLS.

echo "=== MLM All-Layers Extraction (SEAL + ORCC) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/seal_round4/activations/mlm_tier0
mkdir -p v_1/src/linear_probing/results/orcc_round1/activations/mlm_tier0

# --- SEAL corpus (text column = "text") ------------------------------------
echo ""
echo "--- SEAL ---"
python -u v_1/src/archive/baseline_mlm/03_extract_seal_embeddings_all_layers.py \
    --input-parquet v_1/data/evaluation/corpora/seal_corpus.parquet \
    --text-col text \
    --output-dir v_1/src/linear_probing/results/seal_round4/activations/mlm_tier0 \
    || { echo "FAILED: SEAL MLM all-layers extraction"; exit 1; }

# --- ORCC corpus (text column = "text_tier0") -------------------------------
echo ""
echo "--- ORCC ---"
python -u v_1/src/archive/baseline_mlm/03_extract_seal_embeddings_all_layers.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_tier0 \
    --output-dir v_1/src/linear_probing/results/orcc_round1/activations/mlm_tier0 \
    || { echo "FAILED: ORCC MLM all-layers extraction"; exit 1; }

# NOTE: *.npz files are gitignored — activations remain on cluster only.
# If you want to commit the metadata files:
#   git add v_1/src/linear_probing/results/seal_round4/activations/mlm_tier0/metadata.json
#   git add v_1/src/linear_probing/results/orcc_round1/activations/mlm_tier0/metadata.json
#   git commit -m "Add MLM all-layers metadata (cluster job $SLURM_JOB_ID)"
#   git push origin main

echo ""
echo "=== Done ==="
echo "End: $(date)"
