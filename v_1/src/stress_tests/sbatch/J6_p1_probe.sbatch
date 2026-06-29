#!/bin/bash
#SBATCH --job-name=J6_p1_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J6_p1_probe_%j.out
# ===========================================================================
# J6 — P1 Gurnee-Tegmark year-probe (CPU). RUN AFTER J4 (needs king activations).
# year @ mean(tier0,maximal) + king_last + king_mean, GroupKFold-by-ruler, MLP-vs-
# linear, proximity. Loops all methods (king_* skipped where J4 didn't produce them).
#   sbatch v_1/src/stress_tests/sbatch/J6_p1_probe.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J6 P1 year-probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/p1_gurnee_tegmark/probe_p1.py --method "$M" || echo "WARN p1 $M failed"
done
git add v_1/src/stress_tests/p1_gurnee_tegmark/results/*.json 2>/dev/null || true
git commit -m "J6: P1 year-probe (job ${SLURM_JOB_ID})" || echo "nothing to commit"
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || true
git push origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN push failed"
echo "=== Done. $(date) ==="
