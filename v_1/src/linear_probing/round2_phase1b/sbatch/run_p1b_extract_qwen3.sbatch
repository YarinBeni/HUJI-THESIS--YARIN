#!/bin/bash
#SBATCH --job-name=r3_p1b_extract_qwen3
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_p1b_extract_qwen3_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# C13 (Round-3) stage 1 — extract PROMPTED activations for pv0..pv3 on the qwen3
# model set, reusing the APPROVED Phase-1b prompts. Job ARRAY: one task per model.
#   task 0 → qwen3_1b7  (Qwen/Qwen3-1.7B)  layers 0,7,14,21,27
#   task 1 → qwen3_8b   (Qwen/Qwen3-8B)    layers 0,9,18,27,36
#   task 2 → qwen3_32b  (Qwen/Qwen3-32B)   layers 0,16,32,48,63
# (per-model layer sweep ≈ {input, 4 evenly-spaced, top}; matched in reprobe.)
#
# Prompts read from the FIXED approved dir and SHA-256-verified before generation.
# NPZs land under a PER-MODEL out_dir so qwen2.5 activations are untouched:
#   v_1/src/linear_probing/results/orcc_round2_phase1b_<MODEL>/prompted_activations/
# NPZs are gitignored; this stage commits only the per-variant metadata.json.
# The reprobe stage (run_p1b_reprobe_qwen3.sbatch) is the terminal committer.
#
#   sbatch v_1/src/linear_probing/round2_phase1b/sbatch/run_p1b_extract_qwen3.sbatch
# ===========================================================================

set -uo pipefail

MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
LAYERSETS=("0,7,14,21,27" "0,9,18,27,36" "0,16,32,48,63")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
HFID="${HFIDS[$SLURM_ARRAY_TASK_ID]}"
LAYERS="${LAYERSETS[$SLURM_ARRAY_TASK_ID]}"

echo "=== C13 stage1 extract — ${MODEL} (${HFID}) layers=${LAYERS} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
PROMPT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1b/prompts
OUT_ROOT=v_1/src/linear_probing/results/orcc_round2_phase1b_${MODEL}
mkdir -p "$OUT_ROOT/prompted_activations"

# ── SHA-256 verify prompts against APPROVED.md ───────────────────────────────
echo ""; echo "--- verify prompt hashes ---"
declare -A EXPECTED_HASH
EXPECTED_HASH[pv0.md]="624708e3e66a31709b244d2f8ddc77d7556628be31c70e40f30bb9ee043e4d01"
EXPECTED_HASH[pv1.md]="1413becf49cdb9cdcf3a3b66135b821871a73d78b8d38e3c9aab8653c80cfa78"
EXPECTED_HASH[pv2.md]="f6397dcd92cb9dadabe9c8e3ce9e8994e440e9f3e17fd5cca02a801e0f2f57df"
EXPECTED_HASH[pv3.md]="db306a0f4cce10dfbc0acb5fe182db11d72d9e004ddec4d106340082788fd5cd"
for f in pv0.md pv1.md pv2.md pv3.md; do
    actual=$(sha256sum "$PROMPT_DIR/$f" | awk '{print $1}')
    if [ "$actual" != "${EXPECTED_HASH[$f]}" ]; then
        echo "FAILED: prompt hash mismatch for $f (expected ${EXPECTED_HASH[$f]}, got $actual)"
        echo "Re-approval required (see APPROVED.md)."; exit 1
    fi
    echo "  OK  $f  $actual"
done

# ── extract per variant ──────────────────────────────────────────────────────
for VARIANT in pv0 pv1 pv2 pv3; do
    echo ""; echo "--- extract_prompted_acts.py variant=$VARIANT layers=$LAYERS poolings=last,mean ---"
    python -u v_1/src/linear_probing/round2_phase1b/extract_prompted_acts.py \
        --variant "$VARIANT" \
        --layers "$LAYERS" \
        --model_path "$HFID" \
        --out_dir "$OUT_ROOT" \
        --corpus v_1/data/evaluation/corpora/orcc_corpus.parquet \
        --prompts_dir "$PROMPT_DIR" \
        --draws_matrix v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
        --fragment_order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
        || { echo "FAILED: extract variant=$VARIANT $MODEL"; exit 1; }
done

echo ""; echo "=== commit ${MODEL} extraction metadata (NPZs gitignored) ==="
git add "$OUT_ROOT"/prompted_activations/*/metadata.json
git commit -m "Round-3 C13 stage1: prompted-activation metadata ${MODEL} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || true
git pull --rebase origin main || true
git push origin main || true

echo "=== Done. End: $(date) ==="
