#!/bin/bash
#SBATCH --job-name=J2c_t9_gptoss_kp1
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J2c_t9_gptoss_kp1_%j.out
# ===========================================================================
# J2c — rerun T9 kp1 ONLY for gpt-oss-120B with max_new_tokens=2048. The J2b run
# used 256, so gpt-oss's reasoning channel ate the budget before the final JSON
# (raw output cut mid-list) -> kp1 scored 0 / rescored floor 0.38. This gives the
# true value. Re-runs parse + strict score + the raw-text rescore afterwards.
#   sbatch v_1/src/stress_tests/sbatch/J2c_t9_gptoss_kp1.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
echo "=== J2c T9 kp1@2048 ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

KP=v_1/src/linear_probing/round2_phase1a
OUT=v_1/src/stress_tests/redo_t9_knowledge/direct_kp_${M}
python -u $KP/run_kp.py   --variant kp1 --model_path "$H" --out_dir "$OUT" --max_new_tokens 2048 || { echo "FAIL run"; exit 1; }
python -u $KP/parse_kp.py --variant kp1 --out_dir "$OUT" || { echo "FAIL parse"; exit 1; }
python -u $KP/score_kp.py --variant kp1 --out_dir "$OUT" || { echo "FAIL score"; exit 1; }
python -u v_1/src/stress_tests/eda/rescore_t9_kp1.py || echo "WARN rescore failed"

commit_push "J2c: T9 kp1 gpt-oss @2048 tokens (job ${SLURM_JOB_ID})" \
    "$OUT/raw" "$OUT/parsed/kp1.json" "$OUT/scores/kp1_metrics.json" \
    v_1/src/stress_tests/results/t9_kp1_rescored.json
echo "=== Done. $(date) ==="
