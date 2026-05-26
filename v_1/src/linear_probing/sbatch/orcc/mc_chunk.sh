#!/bin/bash
#SBATCH --job-name=mc_chunk
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=v_1/src/linear_probing/logs/mc_chunk_%x_%j.out
# Templated balanced-MC chunk job. One probe-set (pls+cls+cls_numeric) for ONE
# model over a draw RANGE, parallelized across layers (joblib threading).
#
# Override per submission:
#   --cpus-per-task=N   (becomes --n-jobs; match to layer count)
#   --mem=...
#   --export=MODEL=qwen3_32b,DRAW_START=0,DRAW_END=49
#
# Writes per-draw JSONs only; does NOT commit (avoids git races across the
# parallel chunks). The mc_aggregate.sh job commits once all chunks finish.

set -euo pipefail

# Single-threaded BLAS so N joblib threads × 1 BLAS-thread = N cores (no
# oversubscription). MUST be set before numpy/sklearn import.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round2_phase0/probes

# PROBES defaults to all three for MODEL, but can be overridden via --export
# (e.g. PROBES=qwen_cls_numeric for a targeted Ridge backfill).
PROBES="${PROBES:-${MODEL}_pls,${MODEL}_cls,${MODEL}_cls_numeric}"
NJOBS="${SLURM_CPUS_PER_TASK:-8}"

# Pooling / cleaning selectors (C3 last-token sweep). Default mean/tier0 keeps
# all existing fan-out behaviour byte-for-byte identical. run_mc_probes.py reads
# the matching <model>_<cleaning>_<pooling> activation dirs and tags its output
# JSON filenames so mean and last draws never collide.
POOLING="${POOLING:-mean}"
CLEANING="${CLEANING:-tier0}"

# Distinct output-filename tag for non-default (cleaning,pooling) so last-token /
# maximal draws never overwrite the existing mean/tier0 draws (which use the
# bare `mc_balanced` tag). Default mean/tier0 keeps the historical tag exactly.
TAG="mc_balanced"
[ "$CLEANING" != "tier0" ] && TAG="${TAG}_${CLEANING}"
[ "$POOLING"  != "mean"  ] && TAG="${TAG}_${POOLING}"
METHOD_TAG="${METHOD_TAG:-$TAG}"

echo "=== MC chunk: ${MODEL} draws ${DRAW_START}-${DRAW_END} pool=${POOLING} clean=${CLEANING} tag=${METHOD_TAG} ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  CPUs: ${NJOBS}  Start: $(date)"
echo "Probes: ${PROBES}"

python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    --fragment-order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    --output-dir     v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes         "$PROBES" \
    --layers         all \
    --draws-range    "${DRAW_START}-${DRAW_END}" \
    --pooling        "$POOLING" \
    --cleaning       "$CLEANING" \
    --method-tag     "$METHOD_TAG" \
    --n-jobs         "$NJOBS" \
    || { echo "FAILED: ${MODEL} chunk ${DRAW_START}-${DRAW_END} (${POOLING}/${CLEANING})"; exit 1; }

echo "=== Chunk done: ${MODEL} ${DRAW_START}-${DRAW_END} ${POOLING}/${CLEANING}  End: $(date) ==="
