#!/bin/bash
#SBATCH --job-name=pls_mlm
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/linear_probing/logs/pls_mlm_%j.out

# CPU-only: PLS on MLM activations (d_model=384, 17 layers, 893 labeled rows).
# Prerequisites: extract_mlm_all_layers.sh must have completed for both corpora.

echo "=== PLS Sweep — MLM (all layers, tier0, mean pooling) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round1/pls

python -u v_1/src/linear_probing/05_compute_pls_mlm.py \
    --cleaning tier0 \
    --pooling mean \
    --layers all \
    --year-transforms raw,log \
    --n-components 1,2,3,5 \
    || { echo "FAILED: PLS MLM sweep"; exit 1; }

echo ""
echo "=== Pushing results to GitHub ==="
git add v_1/src/linear_probing/results/orcc_round1/pls/pls_results_mlm.json \
        v_1/src/linear_probing/results/orcc_round1/pls/pls_projections_mlm.json
git commit -m "Add MLM PLS results + projections (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo ""
echo "=== Done ==="
echo "End: $(date)"
