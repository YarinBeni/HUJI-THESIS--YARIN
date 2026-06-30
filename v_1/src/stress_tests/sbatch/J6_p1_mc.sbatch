#!/bin/bash
#SBATCH --job-name=J6_p1_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J6_p1_mc_%j.out
# ===========================================================================
# J6 (MC) — P1 year-probe under Monte-Carlo BALANCED draws (comparable to the
# maximal-balanced PLS 0.41 headline). mean(tier0,maximal) + king_last/king_mean
# (tier0 only). CPU; reads on-disk acts incl. king acts from J4. Run after J4.
#   sbatch v_1/src/stress_tests/sbatch/J6_p1_mc.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J6 MC P1 year-probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    echo "--- P1 MC: $M ---"
    python -u v_1/src/stress_tests/p1_gurnee_tegmark/probe_p1_mc.py --method "$M" --n-jobs 32 \
        || echo "WARN p1_mc $M failed"
done
git add v_1/src/stress_tests/p1_gurnee_tegmark/results/mc/*.json 2>/dev/null || true
git commit -m "J6 MC: P1 balanced-MC year-probe (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done. $(date) ==="
