#!/bin/bash
#SBATCH --job-name=bias_check
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=bias_check_%j.out

echo "=== Bias Check Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/lititure-review
git pull origin main

# Install/update dependencies
pip install -r v_1/requirements.txt --quiet 2>&1 | tail -5
echo "Environment ready."

# --- Run pipeline (fail-fast) ---
echo "--- Step 1/5: Featurize ---"
python v_1/src/bias_check/01_featurize.py || { echo "FAILED: featurize"; exit 1; }

echo "--- Step 2/5: Train ---"
python v_1/src/bias_check/02_train.py     || { echo "FAILED: train"; exit 1; }

echo "--- Step 3/5: Evaluate ---"
python v_1/src/bias_check/03_evaluate.py  || { echo "FAILED: evaluate"; exit 1; }

echo "--- Step 4/5: Plot ---"
python v_1/src/bias_check/04_plot.py      || { echo "FAILED: plot"; exit 1; }

echo "--- Step 5/5: Report ---"
python v_1/src/bias_check/05_report.py    || { echo "FAILED: report"; exit 1; }

# --- Push results back to GitHub ---
echo "--- Pushing results to GitHub ---"
git add v_1/data/evaluation/bias_check/
git commit -m "Add bias check results (cluster job $SLURM_JOB_ID)" || echo "Nothing to commit"
git push origin main || echo "WARNING: git push failed — results saved locally at v_1/data/evaluation/bias_check/"

echo "=== Pipeline Complete ==="
echo "End: $(date)"
