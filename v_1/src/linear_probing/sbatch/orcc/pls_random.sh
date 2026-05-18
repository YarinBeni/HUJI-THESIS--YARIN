#!/bin/bash
#SBATCH --job-name=orcc_pls_random
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_pls_random_%j.out

echo "=== ORCC PLS — Random (all 4 cleaning × pooling combos) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__probe_pls

# ── year regression (fresh run with global-shuffle baseline) ──────────────────
python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning tier0 --pooling mean \
    --layers all --year-transforms raw,log --n-components 1,2,3,5 \
    --target year --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random tier0 mean year"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning tier0 --pooling last \
    --layers all --year-transforms raw,log --n-components 1,2,3,5 \
    --target year --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random tier0 last year"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning maximal --pooling mean \
    --layers all --year-transforms raw,log --n-components 1,2,3,5 \
    --target year --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random maximal mean year"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning maximal --pooling last \
    --layers all --year-transforms raw,log --n-components 1,2,3,5 \
    --target year --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random maximal last year"; exit 1; }

# ── ruler PLS-DA ──────────────────────────────────────────────────────────────
python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning tier0 --pooling mean \
    --layers all --n-components 1,2,3,5 \
    --target ruler --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random tier0 mean ruler"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning tier0 --pooling last \
    --layers all --n-components 1,2,3,5 \
    --target ruler --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random tier0 last ruler"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning maximal --pooling mean \
    --layers all --n-components 1,2,3,5 \
    --target ruler --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random maximal mean ruler"; exit 1; }

python -u v_1/src/linear_probing/05_compute_pls.py \
    --method random --cleaning maximal --pooling last \
    --layers all --n-components 1,2,3,5 \
    --target ruler --overwrite \
    --output-dir v_1/src/linear_probing/results/orcc__probe_pls \
    || { echo "FAILED: random maximal last ruler"; exit 1; }

echo "=== Pushing results to GitHub ==="
git add \
    v_1/src/linear_probing/results/orcc__probe_pls/pls_results_random.json \
    v_1/src/linear_probing/results/orcc__probe_pls/pls_projections_random.json
git commit -m "Add ORCC PLS results+projections: random year(global-shuffle)+ruler PLS-DA (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally at results/orcc__probe_pls/"

echo "=== Done ==="
echo "End: $(date)"
