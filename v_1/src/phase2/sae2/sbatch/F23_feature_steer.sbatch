#!/bin/bash
#SBATCH --job-name=F23_fsteer
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/phase2/sae2/logs/F23_fsteer_%j.out
# ===========================================================================
# F23 — SAE2 step 5: feature-level interventions (Golden-Gate-style) with the
# Feldman non-surgicality control. Amplify/suppress + ablate on cell-A
# prompts against firing-rate-matched random features, and THE BRIDGE:
# clamp temporal features on mid-text in English glosses — does the signal
# now reach the last token? Submit with --dependency=afterok:<F22 jobid>.
# Inherited caution: F12's direction-steering was null; a null here WITH the
# control is the publishable outcome — no retry-until-significant.
# ===========================================================================
set -uo pipefail
echo "=== F23 feature steer === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae2/logs

python -u v_1/src/phase2/sae2/feature_steer.py || { echo "FAILED steer"; exit 1; }

commit_push "F23: SAE2 feature interventions (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae2/results
echo "=== Done. $(date) ==="
