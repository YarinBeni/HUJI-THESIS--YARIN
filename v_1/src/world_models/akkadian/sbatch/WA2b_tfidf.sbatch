#!/bin/bash
#SBATCH --job-name=WA2b_tfidf
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA2b_tfidf_%j.out
# ===========================================================================
# WA2b — TF-IDF floor for the Akkadian mimic (CPU, no extraction needed).
#   sbatch v_1/src/world_models/akkadian/sbatch/WA2b_tfidf.sbatch
# ===========================================================================
set -uo pipefail
echo "=== WA2b tfidf === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/akkadian/tfidf_akk.py \
    || { echo "FAILED tfidf"; exit 1; }
commit_push "WA2b: akk tfidf floor (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/akkadian/results/probes/tfidf
echo "=== Done. $(date) ==="
