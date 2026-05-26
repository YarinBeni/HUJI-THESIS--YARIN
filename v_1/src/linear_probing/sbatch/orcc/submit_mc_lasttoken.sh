#!/bin/bash
# C3 — balanced-MC LAST-TOKEN fan-out for qwen3_1b7/8b/32b + thalesian_akk300m
# + thalesian_cunei400m + random, all readouts (pls + cls + cls_numeric).
#
# Clone of submit_mc_fanout.sh with --export POOLING=last,CLEANING=tier0. Each
# model gets N_CHUNKS chunk jobs (writing per-draw JSONs, NO commit) + one
# aggregate job (afterok on its chunks) that rebuilds summaries and commits.
#
# Run from repo root on the cluster:
#   bash v_1/src/linear_probing/sbatch/orcc/submit_mc_lasttoken.sh
#
# random depends on C1 (random_tier0_last activations). Pass the C1 job id:
#   C1_JOBID=12345 bash v_1/src/linear_probing/sbatch/orcc/submit_mc_lasttoken.sh
# (Without C1_JOBID, random is skipped with a warning so the other 5 still go.)
#
# PREREQUISITE last-token activation dirs (must already be on disk):
#   qwen3_1b7_tier0_last  qwen3_8b_tier0_last  qwen3_32b_tier0_last
#   thalesian_akk300m_tier0_last  thalesian_cunei400m_tier0_last  (all present)
#   random_tier0_last  <- produced by C1 (extract_random_qwen3_8b.sh)

set -euo pipefail
cd ~/projects/HUJI-THESIS--YARIN

CHUNK=v_1/src/linear_probing/sbatch/orcc/mc_chunk.sh
AGG=v_1/src/linear_probing/sbatch/orcc/mc_aggregate.sh

POOLING=last
CLEANING=tier0

N_DRAWS=200
N_CHUNKS=4
CHUNK_SIZE=$(( N_DRAWS / N_CHUNKS ))   # 50

# Per-model CPUs (≈ layer count) and memory.
declare -A NCPUS=( [qwen3_1b7]=32 [qwen3_8b]=48 [qwen3_32b]=64 \
                   [thalesian_akk300m]=14 [thalesian_cunei400m]=14 [random]=48 )
declare -A MEM=(   [qwen3_1b7]=32G [qwen3_8b]=48G [qwen3_32b]=96G \
                   [thalesian_akk300m]=24G [thalesian_cunei400m]=24G [random]=48G )

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m random)
fi

C1_JOBID="${C1_JOBID:-}"

for MODEL in "${MODELS[@]}"; do
    # random chunks must wait on the C1 extraction (random_tier0_last).
    DEP_ARG=()
    if [ "$MODEL" = "random" ]; then
        if [ -z "$C1_JOBID" ]; then
            echo "WARNING: skipping random (no C1_JOBID given; rerun with C1_JOBID=<id>)"
            continue
        fi
        DEP_ARG=(--dependency="afterok:${C1_JOBID}")
    fi

    echo "=== C3 last-token fan-out ${MODEL}: ${N_CHUNKS} chunks × ${CHUNK_SIZE} draws, ${NCPUS[$MODEL]} CPUs ==="
    CHUNK_JOBS=()
    for (( c=0; c<N_CHUNKS; c++ )); do
        START=$(( c * CHUNK_SIZE ))
        END=$(( c * CHUNK_SIZE + CHUNK_SIZE - 1 ))
        JID=$(sbatch --parsable \
            "${DEP_ARG[@]}" \
            --job-name="mclast_${MODEL}_${START}_${END}" \
            --cpus-per-task="${NCPUS[$MODEL]}" \
            --mem="${MEM[$MODEL]}" \
            --export=ALL,MODEL="$MODEL",DRAW_START="$START",DRAW_END="$END",POOLING="$POOLING",CLEANING="$CLEANING" \
            "$CHUNK")
        echo "  chunk ${START}-${END} → job ${JID}"
        CHUNK_JOBS+=("$JID")
    done
    DEP=$(IFS=:; echo "${CHUNK_JOBS[*]}")
    AGG_JID=$(sbatch --parsable \
        --job-name="mclast_agg_${MODEL}" \
        --dependency="afterok:${DEP}" \
        --export=ALL,MODEL="$MODEL",POOLING="$POOLING",CLEANING="$CLEANING" \
        "$AGG")
    echo "  aggregate (afterok:${DEP}) → job ${AGG_JID}"
    echo ""
done

echo "Monitor: squeue -u \$USER"
