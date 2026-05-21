#!/bin/bash
#SBATCH --job-name=r2_p3_probe_thalesian
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p3_probe_thalesian_%j.out

# ============================================================================
# Round 2 Phase 3 — CLS + PLS probes for both Thalesian models, all configs.
#
# Reads the Thalesian encoder activations extracted by extract_enc_activations.py
# (committed via jobs 8223-8226) and runs the standard Round-1 CLS + PLS probes
# on each (model, cleaning, pooling) combo. Output JSONs are written to the
# Round-1 result dirs with method tags `thalesian_akk300m` / `thalesian_cunei400m`.
#
# After this finishes, 06_aggregate_cls.py / 06_aggregate_pls.py will pick up
# the new rows once the METHODS list is extended (see same commit).
# ============================================================================

echo "=== Round 2 Phase 3 — Thalesian CLS+PLS probes ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__probe_cls
mkdir -p v_1/src/linear_probing/results/orcc__probe_pls

MODELS=(thalesian_akk300m thalesian_cunei400m)
CLEANINGS=(tier0 maximal)
POOLINGS=(mean last)

for model in "${MODELS[@]}"; do
    for cleaning in "${CLEANINGS[@]}"; do
        for pooling in "${POOLINGS[@]}"; do
            echo ""
            echo "--- $model / $cleaning / $pooling ---"
            python -u v_1/src/linear_probing/round2_phase3/probe_thalesian.py \
                --method   "$model" \
                --cleaning "$cleaning" \
                --pooling  "$pooling" \
                --target   both \
                || { echo "FAILED: $model / $cleaning / $pooling"; exit 1; }
        done
    done
done

# Auto-push results back to the repo
git add v_1/src/linear_probing/results/orcc__probe_cls/cls_results_thalesian_*.json \
        v_1/src/linear_probing/results/orcc__probe_pls/pls_results_thalesian_*.json
git commit -m "Phase 3: Thalesian CLS+PLS probe results (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo ""
echo "=== Done ==="
echo "End: $(date)"
