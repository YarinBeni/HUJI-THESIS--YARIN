#!/bin/bash
#SBATCH --job-name=J5c_p3_anchors_random
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J5c_p3_anchors_random_%j.out
# ===========================================================================
# J5c — P3 anchor extraction for the RANDOM-weight model (Qwen3-8B arch,
# from_config, seed=42). Adds random-Qwen3-8B to the P3 timeline ladder
# (user request): the control curve for "do anchors form an ordered line".
#   sbatch v_1/src/stress_tests/sbatch/J5c_p3_anchors_random.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J5c P3 anchors random === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
git fetch origin main && git merge --ff-only FETCH_HEAD || echo "WARN sync failed"
mkdir -p v_1/src/stress_tests/logs

python -u v_1/src/stress_tests/p3_matter_of_time/extract_anchor_acts.py \
    --hfid Qwen/Qwen3-8B --method random --arch causal --random \
    || { echo "FAIL anchors random"; exit 1; }

AN=v_1/src/stress_tests/p3_matter_of_time/anchors/random
git add "$AN/anchors.json" "$AN/metadata.json" 2>/dev/null || true
git commit -m "J5c: P3 anchors random (control) (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done random anchors. $(date) ==="
