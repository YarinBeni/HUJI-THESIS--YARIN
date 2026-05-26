#!/bin/bash
#SBATCH --job-name=r3_phase_c_bal
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/geodesic/results/phase_c/logs/balanced_loro_%j.log
# C11 (Round-3) — BALANCED LORO over the 200 balanced draws on the best configs.
# Single sequential job + single committer.
#
# WHY ONE JOB (not a fan-out): loro.py:run_balanced writes a single fixed-name
# file loro_robustness_balanced.json (under v_1/src/geodesic/results/, IGNORING
# --output-dir) and overwrites it with only the CURRENT config's 1-row summary.
# Parallel jobs would clobber each other. So we run each config sequentially,
# snapshot its 1-row file, then merge all snapshots into the canonical file once.
#
# Configs (best LORO configs incl. the new qwen3_1b7 flagship):
#   qwen3_1b7            / tier0   / mean / L1   (new flagship; pairs with C8)
#   qwen                / maximal / mean / L1    (Phase B overall winner)
#   thalesian_cunei400m / maximal / mean / L7    (best Thalesian)
#   thalesian_cunei400m / tier0   / mean / L6    (Round-2 reference)
# (random is N/A for LORO per the status matrix.)
#
# Independent of C1 — submit immediately.
#   sbatch v_1/src/geodesic/phase_c/sbatch/balanced_loro_job.sh

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/geodesic/results/phase_c/logs
SNAP_DIR=v_1/src/geodesic/results/phase_c/balanced_snapshots
mkdir -p "$SNAP_DIR"

DRAWS=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy
ORDER=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json
CANON=v_1/src/geodesic/results/loro_robustness_balanced.json

echo "=== C11 balanced LORO ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

# config tuples: METHOD CLEANING POOL LAYER
CONFIGS=(
    "qwen3_1b7 tier0 mean 1"
    "qwen maximal mean 1"
    "thalesian_cunei400m maximal mean 7"
    "thalesian_cunei400m tier0 mean 6"
)

for CFG in "${CONFIGS[@]}"; do
    read -r METHOD CLEANING POOL LAYER <<< "$CFG"
    echo ""; echo "--- balanced LORO $METHOD / $CLEANING / $POOL / L$LAYER ---"
    python -u v_1/src/geodesic/phase_c/loro.py \
        --method   "$METHOD" \
        --cleaning "$CLEANING" \
        --pool     "$POOL" \
        --layer    "$LAYER" \
        --draws-matrix   "$DRAWS" \
        --fragment-order "$ORDER" \
        --draw-range 0-199 \
        || { echo "FAILED: balanced LORO $METHOD / $CLEANING / $POOL / L$LAYER"; exit 1; }
    # snapshot this config's 1-row file before the next run overwrites it.
    cp "$CANON" "$SNAP_DIR/${METHOD}_${CLEANING}_${POOL}_L${LAYER}.json"
done

# Merge all snapshots into the canonical balanced LORO file.
echo ""; echo "--- merging balanced LORO snapshots ---"
python - "$SNAP_DIR" "$CANON" <<'PY'
import json, sys
from pathlib import Path
snap, canon = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for f in sorted(snap.glob("*.json")):
    data = json.loads(f.read_text())
    rows.extend(data if isinstance(data, list) else [data])
canon.write_text(json.dumps(rows, indent=2))
print(f"Merged {len(rows)} configs → {canon}")
PY

git add v_1/src/geodesic/results/loro_robustness_balanced.json \
        v_1/src/geodesic/results/phase_c/balanced_snapshots/
git commit -m "Round-3 C11: balanced LORO (4 configs, job $SLURM_JOB_ID)" || true
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
