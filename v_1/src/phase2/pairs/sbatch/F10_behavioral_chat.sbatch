#!/bin/bash
#SBATCH --job-name=F10_behchat
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F10_behchat_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F10 — gap fix G-chat: rerun the behavioural pairwise test WITH the chat
# template (thinking disabled). The v1 No-bias (yes_rate 0-.38) may be a raw
# completion-format artifact on instruct-tuned Qwen3; results write to
# *.chat.json so v1 stays for comparison.
#   0 qwen3_1b7   1 qwen3_8b   2 qwen3_32b
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F10 behavioral chat ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs

FAILED=0
for V in eng_tier0 akk_maximal; do
    python -u v_1/src/phase2/pairs/behavioral_pairs.py --method "$M" \
        --variant "$V" --chat || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F10: behavioral chat-template ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/pairs/results/behavioral
echo "=== Done ${M}. $(date) ==="
