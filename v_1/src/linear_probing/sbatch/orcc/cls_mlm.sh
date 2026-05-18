#!/bin/bash
#SBATCH --job-name=orcc_cls_mlm
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_cls_mlm_%j.out

echo "=== ORCC CLS — MLM (ruler + year classification) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__probe_cls

python -u v_1/src/linear_probing/05_compute_cls_mlm.py \
    || { echo "FAILED: mlm cls"; exit 1; }

echo "=== Pushing results ==="
git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_mlm.json
git commit -m "Add ORCC CLS results: mlm all layers (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main || echo "WARNING: git push failed"

echo "End: $(date)"
