#!/bin/bash
# Submit Phase C LORO jobs for the top 3 configurations from Phase B.
# Run from repo root: bash v_1/src/geodesic/phase_c/sbatch/submit_loro.sh
#
# Configs (from geodesic_best_layers.json):
#   1. qwen / maximal / mean / L1   — Phase B overall winner
#   2. thalesian_cunei400m / maximal / mean / L7   — best Thalesian
#   3. thalesian_cunei400m / tier0  / mean / L6   — Round 2 reference config

SCRIPT=v_1/src/geodesic/phase_c/sbatch/loro_job.sh
JOB_IDS=()

# 11 rulers with >=10 fragments
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

# Config 1: Phase B overall winner
submit_config qwen maximal mean 1

# Config 2: best Thalesian
submit_config thalesian_cunei400m maximal mean 7

# Config 3: Round 2 reference
submit_config thalesian_cunei400m tier0 mean 6

echo ""
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Aggregate when all done:"
echo "  git pull --rebase origin main"
echo "  python v_1/src/geodesic/phase_c/aggregate_loro.py"
