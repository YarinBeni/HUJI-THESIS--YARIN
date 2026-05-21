#!/bin/bash
#SBATCH --job-name=r2_p1b_reprobe_arr
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1b_reprobe_arr_%A_%a.out

# ===========================================================================
# Round 2 Phase 1b — Re-probing as a Slurm job array (parallel by variant)
#
# 4 array tasks (one per pv variant), each does 2 poolings x 6 layers = 12
# (variant, pooling, layer) combos. Each per-combo CLS+PLS is fast on CPU
# (~2 min); a single variant runs in ~25-30 min wall.
#
# All 4 array tasks run IN PARALLEL on separate compute nodes — total wall
# time = max-of-4 (~30 min) instead of sum-of-4 (~2h) like the original
# single-job version.
#
# After this array completes, run run_p1b_reprobe_aggregate.sbatch (with
# --dependency=afterok:<this_array_jobid>) to build the consolidated
# phase1b_summary.json + phase1b_report.md from the 48 per-combo JSONs.
# ===========================================================================

VARIANTS=(pv0 pv1 pv2 pv3)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Round 2 Phase 1b — reprobe array task ==="
echo "Job ID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Variant: $VARIANT"
echo "Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

# Verify the prompted-activation NPZs exist for this variant
ACTS_DIR="v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations/$VARIANT"
if [ ! -d "$ACTS_DIR" ]; then
    echo "FAILED: missing activations dir for $VARIANT: $ACTS_DIR"
    exit 1
fi
echo "Activations dir: $ACTS_DIR  ($(find $ACTS_DIR -name '*.npz' | wc -l) npz files)"

# Run reprobe for THIS variant only (2 poolings x 6 layers = 12 combos).
# reprobe_pv.py is now resumable — if per-combo JSONs already exist they're
# loaded from disk and skipped, so re-running this task is idempotent.
python -u v_1/src/linear_probing/round2_phase1b/reprobe_pv.py \
    --acts_root v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations \
    --corpus    v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --out_dir   v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing \
    --variants  "$VARIANT" \
    --poolings  last,mean \
    --layers    0,4,10,15,22,28 \
    || { echo "FAILED: reprobe_pv.py for variant $VARIANT"; exit 1; }

echo ""
echo "=== Done $VARIANT ==="
echo "End: $(date)"
