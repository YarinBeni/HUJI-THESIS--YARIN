#!/bin/bash
#SBATCH --job-name=J2a_t9_qwen3
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=v_1/src/stress_tests/logs/J2a_t9_qwen3_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# J2a — redo T9 direct-knowledge elicitation (kp0/kp1/kp2) on the qwen3 set,
# reusing the APPROVED prompts + run_kp/parse_kp/score_kp. Array: one per model.
#   0 qwen3_1b7   1 qwen3_8b   2 qwen3_32b
# Fresh out_dir under stress_tests so the prior phase1a results are untouched.
#   sbatch v_1/src/stress_tests/sbatch/J2a_t9_qwen3.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"; H="${HFIDS[$SLURM_ARRAY_TASK_ID]}"
echo "=== J2a T9 ${M} (${H}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs

KP=v_1/src/linear_probing/round2_phase1a
OUT=v_1/src/stress_tests/redo_t9_knowledge/direct_kp_${M}
mkdir -p "$OUT/raw" "$OUT/parsed" "$OUT/scores"
for V in kp0 kp1 kp2; do
    python -u $KP/run_kp.py   --variant "$V" --model_path "$H" --out_dir "$OUT" --max_new_tokens 512 || { echo "FAIL run $V"; exit 1; }
    python -u $KP/parse_kp.py --variant "$V" --out_dir "$OUT" || { echo "FAIL parse $V"; exit 1; }
    python -u $KP/score_kp.py --variant "$V" --out_dir "$OUT" || { echo "FAIL score $V"; exit 1; }
done
git add "$OUT/"
git commit -m "J2a: redo T9 elicitation ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || echo "nothing to commit"
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || true
git push origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN push failed"
echo "=== Done ${M}. $(date) ==="
