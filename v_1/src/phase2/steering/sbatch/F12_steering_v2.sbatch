#!/bin/bash
#SBATCH --job-name=F12_steer2
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/steering/logs/F12_steer2_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# F12 — gap fixes G-layers + G-alpha: the fair steering test the v1 null owed.
#  * blocks across the EARLY-MID stack (the NAACL band), not just around the
#    late probe layer;
#  * --alpha-mode rel: the push scales with the residual norm at the position,
#    so late-layer massive activations can't drown a fixed-size nudge;
#  * chat template on (matches how instruct models were trained to answer).
# Writes *.v2early.json next to the v1 files.
#   0 qwen3_8b/cellA    1 olmo2_7b/cellA
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_8b olmo2_7b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F12 steering v2 ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/steering/logs

EXTRA=""
[ "$M" = "qwen3_8b" ] && EXTRA="--chat"
python -u v_1/src/phase2/steering/steer_pairs.py --method "$M" --cell A \
    --blocks 4 8 12 16 20 --alpha-mode rel --tag v2early $EXTRA \
    || { echo "FAILED steer v2 ${M}"; exit 1; }

commit_push "F12: steering v2 early-blocks rel-alpha ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/steering/results
echo "=== Done ${M}. $(date) ==="
