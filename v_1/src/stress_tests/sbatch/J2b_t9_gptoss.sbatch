#!/bin/bash
#SBATCH --job-name=J2b_t9_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J2b_t9_gptoss_%j.out
# ===========================================================================
# J2b — T9 direct-knowledge elicitation on gpt-oss-120B (the missing model).
# Reuses run_kp/parse_kp/score_kp. Needs gpu:4 (device_map auto-shards).
#   sbatch v_1/src/stress_tests/sbatch/J2b_t9_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
echo "=== J2b T9 ${M} (${H}) === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
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
git commit -m "J2b: T9 elicitation gpt_oss_120b (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done ${M}. $(date) ==="
