#!/bin/bash
#SBATCH --job-name=r3_phase_b_bal
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_b/logs/balanced_scan_%j.log
# C10 (Round-3) — BALANCED geodesic scan over the 200 balanced draws for
# qwen3_1b7/8b/32b + thalesian_akk300m/cunei400m + random, all configs
# (tier0+maximal x mean+last). Single sequential job + single committer.
#
# WHY ONE JOB (not a per-model fan-out): scan.py:run_balanced writes a single
# fixed-name file geodesic_layer_scoreboard_balanced.json and OVERWRITES it with
# only the current run's records (it does not merge). Running models in parallel
# jobs to the same --output-dir would clobber each other. So we run each model
# into its OWN per-model output-dir, then merge the per-model arrays into the
# canonical scoreboard once, and commit once.
#
# random must run AFTER C1 — submit with --dependency=afterok:<C1_JOBID>.
#   sbatch --dependency=afterok:<C1_JOBID> v_1/src/geodesic/phase_b/sbatch/balanced_scan_job.sh
# (If C1 already finished, plain sbatch is fine.)

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/geodesic/results/phase_b/logs
BAL_ROOT=v_1/src/geodesic/results/phase_b/balanced
mkdir -p "$BAL_ROOT"

DRAWS=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy
ORDER=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json

echo "=== C10 balanced geodesic scan ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

MODELS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m random)

for METHOD in "${MODELS[@]}"; do
    for CLEANING in tier0 maximal; do
        for POOL in mean last; do
            OUT="$BAL_ROOT/${METHOD}_${CLEANING}_${POOL}"
            mkdir -p "$OUT"
            echo ""; echo "--- balanced scan $METHOD / $CLEANING / $POOL → $OUT ---"
            python -u v_1/src/geodesic/phase_b/scan.py \
                --method   "$METHOD" \
                --cleaning "$CLEANING" \
                --pool     "$POOL" \
                --draws-matrix   "$DRAWS" \
                --fragment-order "$ORDER" \
                --draw-range 0-199 \
                --output-dir "$OUT" \
                || { echo "FAILED: balanced scan $METHOD / $CLEANING / $POOL"; exit 1; }
        done
    done
done

# Merge every per-model balanced scoreboard into the canonical file.
echo ""; echo "--- merging per-model balanced scoreboards ---"
python - "$BAL_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("*/geodesic_layer_scoreboard_balanced.json")):
    data = json.loads(f.read_text())
    rows.extend(data if isinstance(data, list) else [data])
out = root.parent.parent / "geodesic_layer_scoreboard_balanced.json"
out.write_text(json.dumps(rows, indent=2))
print(f"Merged {len(rows)} records → {out}")
PY

git add v_1/src/geodesic/results/geodesic_layer_scoreboard_balanced.json \
        v_1/src/geodesic/results/phase_b/balanced/
git commit -m "Round-3 C10: balanced geodesic scoreboard (6 models, job $SLURM_JOB_ID)" || true
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
