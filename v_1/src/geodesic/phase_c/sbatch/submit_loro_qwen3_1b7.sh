#!/bin/bash
# C8 (Round-3) — Phase C LORO for the new flagship qwen3_1b7.
# Config: tier0 / mean / L1  (best layer from geodesic_best_layers.json:
#   qwen3_1b7__tier0__mean best_layer=1, isomap pacc≈0.723).
# Reuses loro_job.sh (one job per held-out ruler, each commits its own distinct
# per-ruler JSON — no git race). Run aggregate_loro afterwards.
#
# Run from repo root on the cluster:
#   bash v_1/src/geodesic/phase_c/sbatch/submit_loro_qwen3_1b7.sh
# Independent of C1 — submit immediately.

SCRIPT=v_1/src/geodesic/phase_c/sbatch/loro_job.sh
JOB_IDS=()

# Same 11 rulers (>=10 fragments) as submit_loro.sh.
RULERS=(
    "Ashurbanipal"
    "Sennacherib"
    "Esarhaddon"
    "Sargon II"
    "Nebuchadnezzar II"
    "Tiglath-pileser III"
    "Nabonidus"
    "Sîn-šarru-iškun"
    "Nabopolassar"
    "Shalmaneser V"
    "Nebuchadnezzar I"
)

submit_config() {
    local METHOD=$1 CLEANING=$2 POOL=$3 LAYER=$4
    echo "--- Submitting LORO for ${METHOD} ${CLEANING} ${POOL} L${LAYER} ---"
    for RULER in "${RULERS[@]}"; do
        SLUG=$(echo "$RULER" | tr ' ' '_' | tr '-' '_')
        JID=$(sbatch --parsable \
            --export=METHOD=$METHOD,CLEANING=$CLEANING,POOL=$POOL,LAYER=$LAYER,RULER="$RULER",RULER_SLUG=$SLUG \
            $SCRIPT)
        echo "  Submitted '${RULER}' → job $JID"
        JOB_IDS+=($JID)
    done
}

submit_config qwen3_1b7 tier0 mean 1

echo ""
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Aggregate when all done (rebuilds loro_robustness.json, commits):"
echo "  sbatch --dependency=afterok:<all LORO job ids> v_1/src/geodesic/phase_c/sbatch/aggregate_loro_job.sh"
