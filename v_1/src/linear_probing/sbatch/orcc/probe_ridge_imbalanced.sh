#!/bin/bash
#SBATCH --job-name=r3_ridge_imb
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_ridge_imb_%j.out
# Round-3 wrap-up C4: Ridge-year (cls_numeric) IMBALANCED backfill for the
# encoder baselines that lacked it — mlm + thalesian_akk300m + thalesian_cunei400m.
# CPU only (reads existing activation dirs). Uses round2_phase3/probe_thalesian.py
# --target cls_numeric, the same driver probe_random.sh calls.
#   mlm        : tier0/mean only (masked-LM encoder — last/maximal are N/A).
#   thalesian  : {tier0,maximal} x {mean,last}.
# Independent of C1 — submit immediately.
#   sbatch v_1/src/linear_probing/sbatch/orcc/probe_ridge_imbalanced.sh

echo "=== Round-3 C4: Ridge-year imbalanced backfill (mlm + thalesian x2) ==="
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Start: $(date)"

# Single-threaded BLAS — set before numpy import.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs \
         v_1/src/linear_probing/results/orcc__probe_cls_numeric

# mlm: tier0/mean only.
echo ""; echo "--- mlm / tier0 / mean ---"
python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
    --method mlm --cleaning tier0 --pooling mean --target cls_numeric \
    || { echo "FAILED: mlm / tier0 / mean"; exit 1; }

# thalesian x2: full {tier0,maximal} x {mean,last}.
for method in thalesian_akk300m thalesian_cunei400m; do
    for cleaning in tier0 maximal; do
        for pooling in mean last; do
            echo ""; echo "--- $method / $cleaning / $pooling ---"
            python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
                --method "$method" --cleaning "$cleaning" --pooling "$pooling" \
                --target cls_numeric \
                || { echo "FAILED: $method / $cleaning / $pooling"; exit 1; }
        done
    done
done

git add v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_mlm.json \
        v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_thalesian_akk300m.json \
        v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_thalesian_cunei400m.json
git commit -m "Round-3 C4: Ridge-year imbalanced backfill mlm + thalesian x2 (job $SLURM_JOB_ID)" || true
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="; echo "End: $(date)"
