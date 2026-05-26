#!/bin/bash
# Backfill cls_numeric (Ridge year regression) under balanced MC for the
# baseline methods tfidf / mlm / qwen (Qwen2.5-7B), so the balanced leaderboard
# has Ridge numbers for every method (Thalesian + Qwen3 already done).
#
# Only the cls_numeric probe is run (PLS/CLS baselines already exist from
# Round 2) via the PROBES override on mc_chunk.sh / mc_aggregate.sh.
#
# Run from repo root on the cluster:
#   bash v_1/src/linear_probing/sbatch/orcc/submit_mc_backfill.sh
#
# C5 (Round-3) additions: thalesian_akk300m, thalesian_cunei400m, random are
# now registered. random's chunks must wait on C1 (random_tier0_mean acts) —
# pass its job id:
#   C1_JOBID=12345 bash .../submit_mc_backfill.sh thalesian_akk300m thalesian_cunei400m random

set -euo pipefail
cd ~/projects/HUJI-THESIS--YARIN

CHUNK=v_1/src/linear_probing/sbatch/orcc/mc_chunk.sh
AGG=v_1/src/linear_probing/sbatch/orcc/mc_aggregate.sh

# CPUs ≈ layer count (Ridge is cheap; tfidf has no layers).
declare -A NCPUS=( [tfidf]=4 [mlm]=17 [qwen]=29 \
                   [thalesian_akk300m]=14 [thalesian_cunei400m]=14 [random]=37 )

C1_JOBID="${C1_JOBID:-}"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(tfidf mlm qwen)
fi

for MODEL in "${MODELS[@]}"; do
    PROBE="${MODEL}_cls_numeric"
    # tfidf is single-config (no layers) → one chunk; others → 2 chunks × 100.
    if [ "$MODEL" = "tfidf" ]; then NC=1; CS=200; else NC=2; CS=100; fi

    # random chunks must wait on C1 extraction (random_tier0_mean).
    DEP_ARG=()
    if [ "$MODEL" = "random" ]; then
        if [ -z "$C1_JOBID" ]; then
            echo "WARNING: skipping random (no C1_JOBID given; rerun with C1_JOBID=<id>)"
            continue
        fi
        DEP_ARG=(--dependency="afterok:${C1_JOBID}")
    fi

    echo "=== Backfill ${PROBE}: ${NC} chunk(s) × ${CS} draws, ${NCPUS[$MODEL]} CPUs ==="
    CHUNK_JOBS=()
    for (( c=0; c<NC; c++ )); do
        START=$(( c * CS ))
        END=$(( c * CS + CS - 1 ))
        JID=$(sbatch --parsable \
            "${DEP_ARG[@]}" \
            --job-name="mcbf_${MODEL}_${START}_${END}" \
            --cpus-per-task="${NCPUS[$MODEL]}" \
            --mem=32G \
            --export=ALL,MODEL="$MODEL",PROBES="$PROBE",DRAW_START="$START",DRAW_END="$END" \
            "$CHUNK")
        echo "  chunk ${START}-${END} → job ${JID}"
        CHUNK_JOBS+=("$JID")
    done
    DEP=$(IFS=:; echo "${CHUNK_JOBS[*]}")
    AGG_JID=$(sbatch --parsable \
        --job-name="mcbf_agg_${MODEL}" \
        --dependency="afterok:${DEP}" \
        --export=ALL,MODEL="$MODEL",PROBES="$PROBE" \
        "$AGG")
    echo "  aggregate (afterok:${DEP}) → job ${AGG_JID}"
    echo ""
done

echo "Monitor: squeue -u \$USER"
