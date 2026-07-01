#!/bin/bash
#SBATCH --job-name=J5b_p3_anchors_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J5b_p3_anchors_gptoss_%j.out
# ===========================================================================
# J5b — P3 anchor extraction for gpt-oss-120B (needs gpu:4; device_map=auto in
# extract_lib). Adds gpt-oss to the P3 timeline ladder (user request). Then J8
# picks it up in the timeline loop.
#   sbatch v_1/src/stress_tests/sbatch/J5b_p3_anchors_gptoss.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J5b P3 anchors gpt-oss === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
git fetch origin main && git merge --ff-only FETCH_HEAD || echo "WARN sync failed"
mkdir -p v_1/src/stress_tests/logs

python -u v_1/src/stress_tests/p3_matter_of_time/extract_anchor_acts.py \
    --hfid openai/gpt-oss-120b --method gpt_oss_120b --arch causal \
    || { echo "FAIL anchors gpt_oss_120b"; exit 1; }

AN=v_1/src/stress_tests/p3_matter_of_time/anchors/gpt_oss_120b
git add "$AN/anchors.json" "$AN/metadata.json" 2>/dev/null || true
git commit -m "J5b: P3 anchors gpt_oss_120b (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done gpt-oss anchors. $(date) ==="
