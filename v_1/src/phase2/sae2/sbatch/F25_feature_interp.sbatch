#!/bin/bash
#SBATCH --job-name=F25_finterp
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/phase2/sae2/logs/F25_finterp_%j.out
# ===========================================================================
# F25 — feature interpretation WITHOUT the logit lens (user's objection to
# F24: lens at layer 9/36 is unreliable). Standard practice instead:
# (1) max-activating contexts across our three populations;
# (2) Golden-Gate-style clamped generation with the chat template — what the
#     model starts talking about IS the feature's behavioural label.
# ===========================================================================
set -uo pipefail
echo "=== F25 feature interp === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae2/logs

python -u v_1/src/phase2/sae2/feature_interp.py || { echo "FAILED interp"; exit 1; }

commit_push "F25: SAE2 feature interpretation — max-activating + clamped generation (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae2/results
echo "=== Done. $(date) ==="
