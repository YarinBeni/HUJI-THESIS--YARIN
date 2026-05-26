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
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs \
         v_1/src/linear_probing/results/orcc__probe_cls \
         v_1/src/linear_probing/results/orcc__probe_pls \
         v_1/src/linear_probing/results/orcc__probe_cls_numeric \
         v_1/src/geodesic/results/phase_b/logs

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

# --- Imbalanced geodesic scan for random (matches the qwen3 4-combo sweep) ---
# scan.py writes phase_b_random_<cleaning>_<pool>.json; random is a full
# qwen3-8b-init, so it gets the same tier0+maximal x mean+last sweep the other
# neural models got in phase_b/sbatch/submit_all.sh.
echo ""; echo "=== C2 geodesic scan: random (imbalanced) ==="
for cleaning in tier0 maximal; do
    for pool in mean last; do
        echo ""; echo "--- geodesic scan random / $cleaning / $pool ---"
        python -u v_1/src/geodesic/phase_b/scan.py \
            --method   random \
            --cleaning "$cleaning" \
            --pool     "$pool" \
            --output-dir v_1/src/geodesic/results/phase_b \
            || { echo "FAILED: geodesic scan random / $cleaning / $pool"; exit 1; }
    done
done

git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_random.json \
        v_1/src/linear_probing/results/orcc__probe_pls/pls_results_random.json \
        v_1/src/linear_probing/results/orcc__probe_cls_numeric/cls_numeric_results_random.json \
        v_1/src/geodesic/results/phase_b/phase_b_random_*.json
git commit -m "Round-3 C2: random (qwen3-8b-init) imbalanced CLS+PLS+Ridge + geodesic scan (job $SLURM_JOB_ID)" || true
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="; echo "End: $(date)"
