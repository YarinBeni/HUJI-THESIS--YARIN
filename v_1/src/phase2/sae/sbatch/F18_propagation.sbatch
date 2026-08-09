#!/bin/bash
#SBATCH --job-name=F18_prop
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/sae/logs/F18_prop_%j.out
# ===========================================================================
# F18 — E-prop, the experiment F11 opened: max-pool the candidate year
# features over every token of each fragment, then run the E1 pairwise
# protocol on those features. Beats the floor on English -> the collapse is a
# PROPAGATION failure and feature-pooling partially fixes it. Doesn't -> the
# mid-text firings are temporal noise.
# ===========================================================================
set -uo pipefail
echo "=== F18 propagation === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae/logs

python -u v_1/src/phase2/sae/propagation_probe.py || { echo "FAILED"; exit 1; }

commit_push "F18: propagation probe (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae/results
echo "=== Done. $(date) ==="
