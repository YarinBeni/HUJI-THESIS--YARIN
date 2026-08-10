#!/bin/bash
#SBATCH --job-name=F26_ignite
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/phase2/steering/logs/F26_ignite_%j.out
# ===========================================================================
# F26 — igniting the anchor inside documents (the last unrun experiment,
# formerly "steering in cell C"; re-justified by F23's positive amplify).
# Arm FEAT: clamp top-5 onomastic features at the ruler-NAME token span of
#   English glosses / all-but-last of Akkadian, vs rate-matched controls.
# Arm DIR: alpha_rel * ||h|| * w_hat (cell-A ridge) at blocks 8/16/24 vs a
#   random unit direction. Read-out: frozen cell-A probe at the last token.
# Pre-registered rules in the script docstring; null-with-control is
# publishable — no retry-until-significant.
# ===========================================================================
set -uo pipefail
echo "=== F26 ignite anchor === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/steering/logs

python -u v_1/src/phase2/steering/ignite_anchor.py || { echo "FAILED ignite"; exit 1; }

commit_push "F26: anchor ignition in documents (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/steering/results
echo "=== Done. $(date) ==="
