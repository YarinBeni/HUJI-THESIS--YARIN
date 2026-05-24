#!/bin/bash
#SBATCH --job-name=r3_e1_probe_8b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_probe_8b_%j.out
# Submit with: sbatch --dependency=afterok:8572:8574:8576:8578 probe_qwen3_8b.sh

echo "=== Phase E1: Qwen3-8B CLS+PLS probes ==="
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
        echo "--- qwen3_8b / $cleaning / $pooling ---"
        python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
            --method   qwen3_8b \
            --cleaning "$cleaning" \
            --pooling  "$pooling" \
            --target   both \
            || { echo "FAILED: qwen3_8b / $cleaning / $pooling"; exit 1; }
    done
done

git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_qwen3_8b.json \
        v_1/src/linear_probing/results/orcc__probe_pls/pls_results_qwen3_8b.json
git commit -m "Phase E1: Qwen3-8B CLS+PLS probe results (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo ""
echo "=== Done ==="
echo "End: $(date)"
