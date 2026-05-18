#!/bin/bash
#SBATCH --job-name=orcc_cls_qwen
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_cls_qwen_%j.out

echo "=== ORCC CLS — Qwen (ruler + year classification) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__probe_cls

# tier0 / mean
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method qwen --cleaning tier0 --pooling mean \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc__probe_cls \
    || { echo "FAILED: qwen tier0 mean"; exit 1; }

# tier0 / last
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method qwen --cleaning tier0 --pooling last \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc__probe_cls \
    || { echo "FAILED: qwen tier0 last"; exit 1; }

# maximal / mean
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method qwen --cleaning maximal --pooling mean \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc__probe_cls \
    || { echo "FAILED: qwen maximal mean"; exit 1; }

# maximal / last
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method qwen --cleaning maximal --pooling last \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc__probe_cls \
    || { echo "FAILED: qwen maximal last"; exit 1; }

echo "=== Pushing results ==="
git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_qwen.json
git commit -m "Add ORCC CLS results: qwen all configs (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main || echo "WARNING: git push failed"

echo "End: $(date)"
