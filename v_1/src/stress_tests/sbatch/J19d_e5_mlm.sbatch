#!/bin/bash
#SBATCH --job-name=J19d_e5_mlm
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J19d_e5_mlm_%j.out
# ===========================================================================
# J19d — E5 word-shuffle for the MLM (the roster's one missing E5 row).
# Dedicated sign-level extractor (extract_shuffled_mlm.py) writes the
# shuf/unshuf twins for tier0+maximal, then the standard probe scores them.
#   sbatch v_1/src/stress_tests/sbatch/J19d_e5_mlm.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J19d E5 MLM === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/e5_shuffle
python -u $D/extract_shuffled_mlm.py --cleanings tier0,maximal \
    || { echo "FAIL extract"; exit 1; }
python -u $D/probe_e5_mc.py --method mlm --n-jobs 8 \
    || { echo "FAIL probe"; exit 1; }
commit_push "J19d E5: MLM shuffle control (job ${SLURM_JOB_ID})" \
    "$D/results/e5_mc__mlm.json"
echo "=== Done. $(date) ==="
