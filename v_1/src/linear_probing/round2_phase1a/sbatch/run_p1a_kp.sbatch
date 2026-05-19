#!/bin/bash
#SBATCH --job-name=r2_p1a_kp
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1a_kp_%j.out

# ===========================================================================
# Round 2 Phase 1a — Direct factual Q&A to Qwen (no fragment context)
#   kp0: per-ruler floruit year question (8 queries)
#   kp1: identify-ruler-from-list                  (2 queries)
#   kp2: ruler validity / false-positive trap      (8 queries)
#
# Pipeline (per variant):
#   1. SHA-256 verify prompt against APPROVED.md (one-time, all 3 prompts)
#   2. run_kp.py    → $OUT_DIR/raw/<variant>_<idx:02d>.json (one per input)
#   3. parse_kp.py  → $OUT_DIR/parsed/<variant>.json        (consolidated)
#   4. score_kp.py  → $OUT_DIR/scores/<variant>_metrics.json
#
# Dependencies (must exist before submission):
#   v_1/src/linear_probing/round2_phase1a/run_kp.py    (W2.B)
#   v_1/src/linear_probing/round2_phase1a/parse_kp.py  (W2.B)
#   v_1/src/linear_probing/round2_phase1a/score_kp.py  (W2.B)
#   v_1/src/linear_probing/results/orcc_round2_phase1a/prompts/{kp0,kp1,kp2}.md
#   v_1/src/linear_probing/results/orcc_round2_phase1a/ruler_reigns.json
#
# Model resolution: run_kp.py reads env QWEN_MODEL_PATH then falls back to
#   HF id "Qwen/Qwen2.5-7B-Instruct". Leave QWEN_MODEL_PATH unset to use HF.
# ===========================================================================

echo "=== Round 2 Phase 1a — direct KP Q&A ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
PROMPT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1a/prompts
OUT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1a/direct_kp
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/parsed" "$OUT_DIR/scores"

# ── Step 1: SHA-256 verify prompts against APPROVED.md ───────────────────────
echo ""
echo "--- Step 1: verify prompt hashes ---"
declare -A EXPECTED_HASH
EXPECTED_HASH[kp0.md]="c56fb7b7430f19623aad255a76196732669c767b2246befdb9d6de03ebfe9df2"
EXPECTED_HASH[kp1.md]="f4c599c395d3135b0ed4b892c7e1736c879c66031b154290aae4e510681119f8"
EXPECTED_HASH[kp2.md]="5a5ed76dec4041721222e89f349891cc71ee24879758025f88dcc44aac1e2a47"

for f in kp0.md kp1.md kp2.md; do
    actual=$(sha256sum "$PROMPT_DIR/$f" | awk '{print $1}')
    expected=${EXPECTED_HASH[$f]}
    if [ "$actual" != "$expected" ]; then
        echo "FAILED: prompt hash mismatch for $f"
        echo "  expected: $expected"
        echo "  actual  : $actual"
        echo "Re-approval required (see APPROVED.md)."
        exit 1
    fi
    echo "  OK  $f  $actual"
done

# ── Step 2-4: run / parse / score each variant ───────────────────────────────
# NOTE: All three scripts take only --variant + --out_dir; they discover the
# prompt at PROMPTS_DIR/{variant}.md and lay out files under
# $OUT_DIR/{raw,parsed,scores}/ themselves. Do not pass --prompt or --input.
for VARIANT in kp0 kp1 kp2; do
    echo ""
    echo "--- $VARIANT: run_kp.py ---"
    python -u v_1/src/linear_probing/round2_phase1a/run_kp.py \
        --variant "$VARIANT" \
        --out_dir "$OUT_DIR" \
        --max_new_tokens 512 \
        || { echo "FAILED: run_kp.py --variant $VARIANT"; exit 1; }

    echo "--- $VARIANT: parse_kp.py ---"
    python -u v_1/src/linear_probing/round2_phase1a/parse_kp.py \
        --variant "$VARIANT" \
        --out_dir "$OUT_DIR" \
        || { echo "FAILED: parse_kp.py --variant $VARIANT"; exit 1; }

    echo "--- $VARIANT: score_kp.py ---"
    python -u v_1/src/linear_probing/round2_phase1a/score_kp.py \
        --variant "$VARIANT" \
        --out_dir "$OUT_DIR" \
        || { echo "FAILED: score_kp.py --variant $VARIANT"; exit 1; }
done

# ── Step 5: push results to GitHub ───────────────────────────────────────────
echo ""
echo "=== Pushing Phase 1a results to GitHub ==="
git add "$OUT_DIR/"
git commit -m "Add Round 2 Phase 1a results: direct KP Q&A kp0/kp1/kp2 (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally."

echo ""
echo "=== Done ==="
echo "End: $(date)"
