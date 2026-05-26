#!/bin/bash
#SBATCH --job-name=mc_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=v_1/src/linear_probing/logs/mc_agg_%x_%j.out
# Rebuild the full summary JSONs for one model from ALL 200 draw files, then
# commit + push everything. Submit with --dependency=afterok:<all chunk jobs>.
#   --export=MODEL=qwen3_32b

set -euo pipefail
export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN: git pull failed"

PROBES="${PROBES:-${MODEL}_pls,${MODEL}_cls,${MODEL}_cls_numeric}"
POOLING="${POOLING:-mean}"
CLEANING="${CLEANING:-tier0}"
# Same tag derivation as mc_chunk.sh so the aggregate reads exactly the draws
# the chunks wrote.
TAG="mc_balanced"
[ "$CLEANING" != "tier0" ] && TAG="${TAG}_${CLEANING}"
[ "$POOLING"  != "mean"  ] && TAG="${TAG}_${POOLING}"
METHOD_TAG="${METHOD_TAG:-$TAG}"
echo "=== MC aggregate: ${MODEL} (${PROBES}) pool=${POOLING} clean=${CLEANING} tag=${METHOD_TAG} ==="

# All draw files already exist -> every draw is SKIPPED, only summaries rebuilt
# from the complete 0..199 set.
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    --fragment-order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    --output-dir     v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes         "$PROBES" \
    --layers         all \
    --draws-range    "0-199" \
    --pooling        "$POOLING" \
    --cleaning       "$CLEANING" \
    --method-tag     "$METHOD_TAG" \
    --n-jobs         1 \
    || { echo "FAILED: aggregate ${MODEL} (${POOLING}/${CLEANING})"; exit 1; }

# Report draw-file counts as a completeness check, matched on this run's tag.
for P in pls cls cls_numeric; do
    N=$(ls v_1/src/linear_probing/results/orcc_round2_phase0/probes/${MODEL}_${P}__${METHOD_TAG}__draw*.json 2>/dev/null | wc -l)
    echo "  ${MODEL}_${P} (${METHOD_TAG}): ${N} draw files"
done

git add v_1/src/linear_probing/results/orcc_round2_phase0/probes/${MODEL}_*
git commit -m "Round-3 MC fan-out: ${MODEL} balanced probes ${POOLING}/${CLEANING} (job ${SLURM_JOB_ID})" \
    || echo "Nothing new to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN: git push failed"

echo "=== Aggregate done: ${MODEL} ${POOLING}/${CLEANING}  End: $(date) ==="
