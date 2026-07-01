#!/bin/bash
#SBATCH --job-name=J8_p3_timeline
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J8_p3_timeline_%j.out
# ===========================================================================
# J8 — P3 timeline analysis (CPU). RUN AFTER J5. 3a: do anchors form an ordered
# 1-D timeline (PCA-1D + Isomap-1D vs year)?  3b: do ORCC texts project onto it
# (nearest-anchor -> year)?  High 3a + low 3b = the dissociation, geometrically.
#   sbatch v_1/src/stress_tests/sbatch/J8_p3_timeline.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J8 P3 timeline === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/p3_matter_of_time/timeline_p3.py --method "$M" || echo "WARN p3 $M failed"
done
git add v_1/src/stress_tests/p3_matter_of_time/results/*.json 2>/dev/null || true
git commit -m "J8: P3 timeline analysis (job ${SLURM_JOB_ID})" || echo "nothing to commit"
push_main
echo "=== Done. $(date) ==="
