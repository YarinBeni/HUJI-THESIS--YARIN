#!/bin/bash
#SBATCH --job-name=F8_feat
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/phase2/sae/logs/F8_feat_%j.out
# ===========================================================================
# F8 — E5.2 feature hunt. Runs at the best FVU-gated layer: which Scope
# features track the entity year on cell A, which align with the ridge
# direction, and whether those features fire on Akkadian fragments at all.
# Submit with --dependency=afterok:<F7 jobid> (submit_all.sh does this).
# ===========================================================================
set -uo pipefail
echo "=== F8 feature hunt === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae/logs

python -u v_1/src/phase2/sae/feature_hunt.py || { echo "FAILED hunt"; exit 1; }

commit_push "F8: SAE feature hunt (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae/results
echo "=== Done. $(date) ==="
