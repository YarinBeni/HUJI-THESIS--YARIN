#!/bin/bash
# Fan out the Phase E1 balanced-MC sweep across draw chunks × layer-threads.
# Run from repo root on the cluster:
#   bash v_1/src/linear_probing/sbatch/orcc/submit_mc_fanout.sh
#
# Per model: N_CHUNKS chunk jobs (each a draw range, all 3 probes, layers
# parallelized across --cpus-per-task threads) + 1 aggregation job that waits
# on all chunks (afterok) and commits.
#
# Resumability: draws whose JSON already exists are SKIPPED, so the 200 valid
# qwen3_1b7 PLS draw files already on disk are reused (those chunks only fill
# in cls + cls_numeric).

set -euo pipefail
cd ~/projects/HUJI-THESIS--YARIN

CHUNK=v_1/src/linear_probing/sbatch/orcc/mc_chunk.sh
AGG=v_1/src/linear_probing/sbatch/orcc/mc_aggregate.sh

N_DRAWS=200
N_CHUNKS=4
CHUNK_SIZE=$(( N_DRAWS / N_CHUNKS ))   # 50

# Per-model CPUs (≈ layer count) and memory.
declare -A NCPUS=( [qwen3_1b7]=32 [qwen3_8b]=48 [qwen3_32b]=64 )
declare -A MEM=(   [qwen3_1b7]=32G [qwen3_8b]=48G [qwen3_32b]=96G )

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
fi

for MODEL in "${MODELS[@]}"; do
    echo "=== Fanning out ${MODEL}: ${N_CHUNKS} chunks × ${CHUNK_SIZE} draws, ${NCPUS[$MODEL]} CPUs ==="
    CHUNK_JOBS=()
    for (( c=0; c<N_CHUNKS; c++ )); do
        START=$(( c * CHUNK_SIZE ))
        END=$(( c * CHUNK_SIZE + CHUNK_SIZE - 1 ))
        JID=$(sbatch --parsable \
            --job-name="mc_${MODEL}_${START}_${END}" \
            --cpus-per-task="${NCPUS[$MODEL]}" \
            --mem="${MEM[$MODEL]}" \
            --export=ALL,MODEL="$MODEL",DRAW_START="$START",DRAW_END="$END" \
            "$CHUNK")
        echo "  chunk ${START}-${END} → job ${JID}"
        CHUNK_JOBS+=("$JID")
    done
    DEP=$(IFS=:; echo "${CHUNK_JOBS[*]}")
    AGG_JID=$(sbatch --parsable \
        --job-name="mc_agg_${MODEL}" \
        --dependency="afterok:${DEP}" \
        --export=ALL,MODEL="$MODEL" \
        "$AGG")
    echo "  aggregate (afterok:${DEP}) → job ${AGG_JID}"
    echo ""
done

echo "Monitor: squeue -u \$USER"
