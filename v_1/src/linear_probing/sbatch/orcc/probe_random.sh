#!/bin/bash
#SBATCH --job-name=r3_probe_random
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_probe_random_%j.out
# Round-3 wrap-up C2: imbalanced CLS+PLS+Ridge probes for the random-init Qwen3-8B
# baseline. CPU only (reads the activations C1 extracted). Reads from the
# random_{cleaning}_{pooling} dirs; writes cls/pls/cls_numeric_results_random.json.
# Submit AFTER C1:  sbatch --dependency=afterok:<C1_JOBID> probe_random.sh

echo "=== Round-3 C2: random (qwen3-8b-init) imbalanced probes ==="
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs \
         v_1/src/linear_probing/results/orcc__probe_cls \
         v_1/src/linear_probing/results/orcc__probe_pls \
         v_1/src/linear_probing/results/orcc__probe_cls_numeric

export OMP_NUM_THREADS=1
for cleaning in tier0 maximal; do
    for pooling in mean last; do
        echo ""; echo "--- random / $cleaning / $pooling ---"
        python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
            --method   random \
            --cleaning "$cleaning" \
            --pooling  "$pooling" \
            --target   all \
            || { echo "FAILED: random / $cleaning / $pooling"; exit 1; }
    done
done

git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_random.json \
        v_1/src/linear_probing/results/orcc__probe_pls/pls_results_random.json \
        v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_random.json
git commit -m "Round-3 C2: random (qwen3-8b-init) imbalanced CLS+PLS+Ridge (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="; echo "End: $(date)"
