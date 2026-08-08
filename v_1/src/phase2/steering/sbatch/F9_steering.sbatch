#!/bin/bash
#SBATCH --job-name=F9_steer
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/phase2/steering/logs/F9_steer_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F9 — E2 causal steering (NAACL'25 recipe): add alpha*w at the second
# entity's name token, sweep alpha and layers, equal-norm random control.
# Read-outs: flip rate AND logit(Yes)-logit(No) shift (robust to the No-bias
# F2 exposed). Cell C (steering inside Akkadian fragments) is deferred to the
# king-token integration — see README.
#   0 qwen3_8b/cellA   1 qwen3_8b/cellB   2 olmo2_7b/cellA
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_8b qwen3_8b olmo2_7b)
CELLS=(A B A)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
C="${CELLS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F9 steer ${M} cell${C} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/steering/logs

if ! ls v_1/src/world_models/results/directions/${M}/historical_figure.*.npz >/dev/null 2>&1; then
    echo "[dir] regenerating cell-A direction for ${M} via probe_wm"
    python -u v_1/src/world_models/probe_wm.py --method "$M" \
        --entity-type historical_figure || { echo "FAILED probe_wm"; exit 1; }
fi

python -u v_1/src/phase2/steering/steer_pairs.py --method "$M" --cell "$C" \
    || { echo "FAILED steer ${M} ${C}"; exit 1; }

commit_push "F9: steering ${M} cell${C} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/steering/results
echo "=== Done. $(date) ==="
