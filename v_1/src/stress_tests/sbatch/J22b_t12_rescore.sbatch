#!/bin/bash
#SBATCH --job-name=J22b_t12_rescore
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J22b_t12_rescore_%j.out
# ===========================================================================
# J22b — T12 RESCORE ONLY (CPU, minutes). The scorer gained ruler macro-F1
# over the canonical-8 classes (+ per-draw arrays); re-run it over the raw
# JSONLs already on disk from J22 and commit the updated result JSONs.
#   sbatch v_1/src/stress_tests/sbatch/J22b_t12_rescore.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J22b T12 rescore === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/t12_forced_dating
python -u $D/score_forced.py --all || { echo "FAIL rescore"; exit 1; }
commit_push "J22b: T12 rescore with ruler macro-F1 (job ${SLURM_JOB_ID})" \
    "$D/results/t12f__*.json"
echo "=== Done. $(date) ==="
