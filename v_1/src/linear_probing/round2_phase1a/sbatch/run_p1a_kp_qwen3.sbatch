#!/bin/bash
#SBATCH --job-name=r3_p1a_kp_qwen3
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_p1a_kp_qwen3_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# C12 (Round-3) — Re-run elicitation kp0/kp1/kp2 on the qwen3 model set, reusing
# the APPROVED Phase-1a prompts. Job ARRAY: one task per qwen3 model.
#   task 0 → qwen3_1b7  (Qwen/Qwen3-1.7B)
#   task 1 → qwen3_8b   (Qwen/Qwen3-8B)
#   task 2 → qwen3_32b  (Qwen/Qwen3-32B)
#
# Pipeline per task: run_kp → parse_kp → score_kp, for each of kp0/kp1/kp2.
# Prompts are read from the FIXED approved dir
#   v_1/src/linear_probing/results/orcc_round2_phase1a/prompts/{kp0,kp1,kp2}.md
# and SHA-256-verified against APPROVED.md before any generation.
#
# Outputs go to a PER-MODEL out_dir so qwen2.5 results are not overwritten:
#   v_1/src/linear_probing/results/orcc_round2_phase1a/direct_kp_<MODEL>/
# Each array task commits ONLY its own out_dir (distinct paths → no git race).
# (T9 table is built later in Phase 3 from these scored results.)
#
#   sbatch v_1/src/linear_probing/round2_phase1a/sbatch/run_p1a_kp_qwen3.sbatch
# ===========================================================================

set -uo pipefail

MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
HFID="${HFIDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== C12 Phase-1a elicitation — ${MODEL} (${HFID}) ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
PROMPT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1a/prompts
OUT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1a/direct_kp_${MODEL}
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/parsed" "$OUT_DIR/scores"

# ── Step 1: SHA-256 verify prompts against APPROVED.md ───────────────────────
echo ""; echo "--- Step 1: verify prompt hashes ---"
declare -A EXPECTED_HASH
EXPECTED_HASH[kp0.md]="c56fb7b7430f19623aad255a76196732669c767b2246befdb9d6de03ebfe9df2"
EXPECTED_HASH[kp1.md]="f4c599c395d3135b0ed4b892c7e1736c879c66031b154290aae4e510681119f8"
EXPECTED_HASH[kp2.md]="5a5ed76dec4041721222e89f349891cc71ee24879758025f88dcc44aac1e2a47"
for f in kp0.md kp1.md kp2.md; do
    actual=$(sha256sum "$PROMPT_DIR/$f" | awk '{print $1}')
    if [ "$actual" != "${EXPECTED_HASH[$f]}" ]; then
        echo "FAILED: prompt hash mismatch for $f (expected ${EXPECTED_HASH[$f]}, got $actual)"
        echo "Re-approval required (see APPROVED.md)."; exit 1
    fi
    echo "  OK  $f  $actual"
done

# ── Step 2-4: run / parse / score each variant ───────────────────────────────
for VARIANT in kp0 kp1 kp2; do
    echo ""; echo "--- $VARIANT: run_kp.py (${MODEL}) ---"
    python -u v_1/src/linear_probing/round2_phase1a/run_kp.py \
        --variant "$VARIANT" \
        --model_path "$HFID" \
        --out_dir "$OUT_DIR" \
        --max_new_tokens 512 \
        || { echo "FAILED: run_kp.py $VARIANT $MODEL"; exit 1; }

    echo "--- $VARIANT: parse_kp.py ---"
    python -u v_1/src/linear_probing/round2_phase1a/parse_kp.py \
        --variant "$VARIANT" --out_dir "$OUT_DIR" \
        || { echo "FAILED: parse_kp.py $VARIANT $MODEL"; exit 1; }

    echo "--- $VARIANT: score_kp.py ---"
    python -u v_1/src/linear_probing/round2_phase1a/score_kp.py \
        --variant "$VARIANT" --out_dir "$OUT_DIR" \
        || { echo "FAILED: score_kp.py $VARIANT $MODEL"; exit 1; }
done

# ── Step 5: commit this model's results (distinct path → no race) ────────────
echo ""; echo "=== Pushing ${MODEL} Phase-1a results ==="
git add "$OUT_DIR/"
git commit -m "Round-3 C12: Phase-1a elicitation kp0/kp1/kp2 on ${MODEL} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    || echo "Nothing new to commit"
git pull --rebase origin main || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done. End: $(date) ==="
