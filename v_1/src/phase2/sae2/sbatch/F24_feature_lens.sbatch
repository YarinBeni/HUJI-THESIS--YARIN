#!/bin/bash
#SBATCH --job-name=F24_flens
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/phase2/sae2/logs/F24_flens_%j.out
# ===========================================================================
# F24 — self-made labels: decoder-row logit lens for the top hunt features.
# Neuronpedia hosts only layer 18 of the release (API-confirmed) and layer 18
# fails our FVU gate, so autointerp labels do not exist for the usable
# layer-9 instrument. CPU only.
# ===========================================================================
set -uo pipefail
echo "=== F24 feature lens === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae2/logs

python -u v_1/src/phase2/sae2/lens_features.py || { echo "FAILED lens"; exit 1; }

commit_push "F24: SAE2 decoder-row feature lens (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae2/results
echo "=== Done. $(date) ==="
