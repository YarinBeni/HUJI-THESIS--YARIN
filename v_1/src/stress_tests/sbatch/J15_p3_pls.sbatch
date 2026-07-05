#!/bin/bash
#SBATCH --job-name=J15_p3_pls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J15_p3_pls_%j.out
# ===========================================================================
# J15 — P3 timeline v2 (CPU): PLS(k=3)-supervised anchor timeline + geodesic
# 1-D + interpolated text projection + ruler-F1, per model x cleaning
# (tier0/maximal/maxking) x layer. Needs anchor npz (cluster NFS) + mean acts.
#   sbatch v_1/src/stress_tests/sbatch/J15_p3_pls.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J15 P3 PLS timeline === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/p3_matter_of_time/timeline_p3_pls.py --method "$M" \
        || echo "WARN p3pls $M failed"
done
commit_push "J15: P3 PLS-geodesic timeline (job ${SLURM_JOB_ID})" \
    v_1/src/stress_tests/p3_matter_of_time/results/pls/*.json
echo "=== Done. $(date) ==="
