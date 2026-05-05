#!/bin/bash
#SBATCH --job-name=orcc_cls_random
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_cls_random_%j.out

echo "=== ORCC CLS — Random (ruler + year classification) ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round1/cls

# tier0 / mean
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method random --cleaning tier0 --pooling mean \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc_round1/cls \
    || { echo "FAILED: random tier0 mean"; exit 1; }

# tier0 / last
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method random --cleaning tier0 --pooling last \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc_round1/cls \
    || { echo "FAILED: random tier0 last"; exit 1; }

# maximal / mean
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method random --cleaning maximal --pooling mean \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc_round1/cls \
    || { echo "FAILED: random maximal mean"; exit 1; }

# maximal / last
python -u v_1/src/linear_probing/05_compute_cls.py \
    --method random --cleaning maximal --pooling last \
    --tasks ruler,year --layers all \
    --output-dir v_1/src/linear_probing/results/orcc_round1/cls \
    || { echo "FAILED: random maximal last"; exit 1; }

echo "=== Pushing results ==="
git add v_1/src/linear_probing/results/orcc_round1/cls/cls_results_random.json
git commit -m "Add ORCC CLS results: random all configs (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main || echo "WARNING: git push failed"

echo "End: $(date)"
