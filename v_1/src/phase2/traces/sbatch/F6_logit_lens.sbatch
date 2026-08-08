#!/bin/bash
#SBATCH --job-name=F6_lens
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/phase2/traces/logs/F6_lens_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F6 — E4.4 logit-lens of the year directions (cell-A ridge + E1 pairwise).
# CPU: loads the model weights once for the unembedding matrix, no forward
# passes. If the direction's top tokens are royal names rather than temporal
# vocabulary, the identity-mediation story has intrinsic, training-free support.
#   0 olmo2_7b   1 llama2_7b   2 qwen3_8b
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b llama2_7b qwen3_8b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F6 lens ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/traces/logs

if ! ls v_1/src/world_models/results/directions/${M}/historical_figure.*.npz >/dev/null 2>&1; then
    echo "[dir] regenerating cell-A direction for ${M} via probe_wm"
    python -u v_1/src/world_models/probe_wm.py --method "$M" \
        --entity-type historical_figure || { echo "FAILED probe_wm"; exit 1; }
fi

python -u v_1/src/phase2/traces/logit_lens.py --method "$M" \
    || { echo "FAILED lens ${M}"; exit 1; }

commit_push "F6: logit-lens ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/traces/results
echo "=== Done ${M}. $(date) ==="
