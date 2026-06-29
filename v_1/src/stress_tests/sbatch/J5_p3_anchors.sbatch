#!/bin/bash
#SBATCH --job-name=J5_p3_anchors
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J5_p3_anchors_%A_%a.out
#SBATCH --array=0-5
# ===========================================================================
# J5 — P3 anchor extraction (GPU, fast: ~150 short prompts/model). Array over the
# GPU-light models; gpt-oss via J5b. Embeds ruler/year anchor prompts -> per-layer
# anchor embeddings. J8 (CPU) then fits the timeline + projects ORCC texts.
#   0 qwen3_1b7  1 qwen3_8b  2 qwen3_32b  3 thalesian_akk300m  4 thalesian_cunei400m  5 umt5_base
#   sbatch v_1/src/stress_tests/sbatch/J5_p3_anchors.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B Thalesian/AKK_300m Thalesian/cuneiformBase-400m google/umt5-base)
ARCHS=(causal causal causal encoder encoder encoder)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"; H="${HFIDS[$SLURM_ARRAY_TASK_ID]}"; A="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
echo "=== J5 P3 anchors ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs

python -u v_1/src/stress_tests/p3_matter_of_time/extract_anchor_acts.py \
    --hfid "$H" --method "$M" --arch "$A" || { echo "FAIL anchors ${M}"; exit 1; }

AN=v_1/src/stress_tests/p3_matter_of_time/anchors/${M}
git add "$AN/anchors.json" "$AN/metadata.json" 2>/dev/null || true
git commit -m "J5: P3 anchors ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || echo "nothing to commit"
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || true
git push origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN push failed"
echo "=== Done ${M}. $(date) ==="
