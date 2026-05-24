#!/bin/bash
#SBATCH --job-name=r3_e1_probe_27b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_probe_27b_%j.out
# Submit with: sbatch --dependency=afterok:<t0mn>:<t0lt>:<mxmn>:<mxlt> probe_qwen35_27b.sh

echo "=== Phase E1: Qwen3.5-27B CLS+PLS probes ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__probe_cls
mkdir -p v_1/src/linear_probing/results/orcc__probe_pls

CLEANINGS=(tier0 maximal)
POOLINGS=(mean last)

for cleaning in "${CLEANINGS[@]}"; do
    for pooling in "${POOLINGS[@]}"; do
        echo ""
        echo "--- qwen35_27b / $cleaning / $pooling ---"
        python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
            --method   qwen35_27b \
            --cleaning "$cleaning" \
            --pooling  "$pooling" \
            --target   all \
            || { echo "FAILED: qwen35_27b / $cleaning / $pooling"; exit 1; }
    done
done

git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_qwen35_27b.json \
        v_1/src/linear_probing/results/orcc__probe_pls/pls_results_qwen35_27b.json \
        v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_qwen35_27b.json
git commit -m "Phase E1: Qwen3.5-27B CLS+PLS probe results (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo ""
echo "=== Done ==="
echo "End: $(date)"
