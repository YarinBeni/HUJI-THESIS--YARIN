#!/bin/bash
#SBATCH --job-name=r3_p1b_reprobe_qwen3
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_p1b_reprobe_qwen3_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# C13 (Round-3) stage 2 — re-probe PLS+CLS on the prompted activations from
# stage 1 (run_p1b_extract_qwen3.sbatch). Job ARRAY: one task per qwen3 model.
# CPU only. Each task is the TERMINAL committer for its own model (distinct
# out_dir → no git race). (T10 table is built later in Phase 3.)
#
# Submit AFTER the extract array:
#   sbatch --dependency=afterok:<EXTRACT_ARRAY_JOBID> run_p1b_reprobe_qwen3.sbatch
# (afterok on the whole array id waits for all 3 extract tasks.)
# ===========================================================================

set -uo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
LAYERSETS=("0,7,14,21,27" "0,9,18,27,36" "0,16,32,48,63")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
LAYERS="${LAYERSETS[$SLURM_ARRAY_TASK_ID]}"

echo "=== C13 stage2 reprobe — ${MODEL} layers=${LAYERS} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
OUT_ROOT=v_1/src/linear_probing/results/orcc_round2_phase1b_${MODEL}
ACTS=$OUT_ROOT/prompted_activations
REPROBE=$OUT_ROOT/reprobing
mkdir -p "$REPROBE"

# verify activations exist
echo ""; echo "--- verify activations ---"
for v in pv0 pv1 pv2 pv3; do
    d="$ACTS/$v"
    [ -d "$d" ] || { echo "FAILED: missing activations dir $d (run extract stage first)"; exit 1; }
    echo "  $v: $(ls "$d"/*.npz 2>/dev/null | wc -l) npz files"
done

echo ""; echo "--- reprobe_pv.py ---"
python -u v_1/src/linear_probing/round2_phase1b/reprobe_pv.py \
    --acts_root "$ACTS" \
    --corpus    v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --out_dir   "$REPROBE" \
    --variants  pv0,pv1,pv2,pv3 \
    --poolings  last,mean \
    --layers    "$LAYERS" \
    || { echo "FAILED: reprobe_pv.py $MODEL"; exit 1; }

echo ""; echo "=== commit ${MODEL} reprobing results ==="
git add "$REPROBE/"
git commit -m "Round-3 C13 stage2: pv0-3 reprobe PLS+CLS ${MODEL} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || true
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
