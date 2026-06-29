#!/bin/bash
#SBATCH --job-name=J9_p7_ksparse
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J9_p7_ksparse_%j.out
# ===========================================================================
# J9 — P7 k-sparse localization (CPU). Runs NOW on existing mean activations.
# How many neurons to recover before/after-median-year? Loops all methods.
#   sbatch v_1/src/stress_tests/sbatch/J9_p7_ksparse.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J9 P7 k-sparse === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/p7_ksparse/probe_p7.py --method "$M" || echo "WARN p7 $M failed"
done
git add v_1/src/stress_tests/p7_ksparse/results/*.json 2>/dev/null || true
git commit -m "J9: P7 k-sparse probe (job ${SLURM_JOB_ID})" || echo "nothing to commit"
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || true
git push origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN push failed"
echo "=== Done. $(date) ==="
