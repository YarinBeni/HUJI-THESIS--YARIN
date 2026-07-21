#!/bin/bash
#SBATCH --job-name=W2b_tfidf
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/W2b_tfidf_%j.out
# ===========================================================================
# W2b — TF-IDF surface-form floor (CPU, no extraction needed). Any embedding arm
# that doesn't clear this is reading substrings, not world structure.
#   sbatch v_1/src/world_models/sbatch/W2b_tfidf.sbatch
# ===========================================================================
set -uo pipefail
echo "=== W2b tfidf === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/tfidf_baseline.py \
    || { echo "FAILED tfidf"; exit 1; }

commit_push "W2b: world-models tfidf baseline (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/results/probes/tfidf
echo "=== Done. $(date) ==="
